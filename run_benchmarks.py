import torch
import os
import json
import argparse
from PIL import Image
from tqdm import tqdm
from huggingface_hub import login
from transformers import CLIPTextModel, T5EncoderModel
from diffusers import AutoencoderKL
from diffusers.models import FluxTransformer2DModel
from slideredit.pipelines import SliderEditFluxKontextPipeline, LoRAAdapterType

# for debugging purposes
class MockPipeline:
    def __call__(self, image, prompt, generator, slider_alpha):
        # Returns a dummy object mimicking diffusers pipeline output: output.images[0]
        class DummyOutput:
            def __init__(self, img):
                # Simply return a copy of the input image (or a blank thumbnail)
                self.images = [img.copy()]
        return DummyOutput(image)

def load_dataset(mapping_file_path, images_dir, dataset_type, start_idx=0, end_idx=None):
    """
    Parses JSON mapping files based on the dataset type and slices the workload.
    """
    print(f"Loading {dataset_type} mapping from {mapping_file_path}...")
    with open(mapping_file_path, 'r') as f:
        mapping_data = json.load(f)
        
    dataset = []
    
    # 1. Parsing Logic
    if dataset_type == "pie-bench":
        # PIE-Bench: {"image_name": {"editing_instruction": "...", ...}}
        for image_key, meta in mapping_data.items():
            base_id, ext = os.path.splitext(image_key)
            # Default to .jpg if no extension provided in the key
            img_filename = image_key if ext else f"{image_key}.jpg"
            clean_id = base_id if ext else image_key
            
            prompt = meta.get("editing_instruction", meta.get("prompt", ""))
            dataset.append({"id": clean_id, "filename": img_filename, "prompt": prompt})
            
    elif dataset_type == "rs-objects":
        # Handle List format: [{"image": "01.jpg", "text": "make it red"}, ...]
        if isinstance(mapping_data, list):
            for item in mapping_data:
                img_filename = item.get("image", item.get("image_id", ""))
                prompt = item.get("text", item.get("prompt", ""))
                clean_id = os.path.splitext(os.path.basename(img_filename))[0]
                dataset.append({"id": clean_id, "filename": img_filename, "prompt": prompt})
                
        # Handle Dict format: {"01.jpg": "make it red", ...}
        elif isinstance(mapping_data, dict):
            for img_filename, prompt in mapping_data.items():
                clean_id = os.path.splitext(os.path.basename(img_filename))[0]
                dataset.append({"id": clean_id, "filename": img_filename, "prompt": prompt})

    # 2. Slicing Logic for Multi-GPU
    sliced_dataset = dataset[start_idx:end_idx] if end_idx is not None else dataset[start_idx:]
    print(f"Processing slice: indices {start_idx} to {start_idx + len(sliced_dataset)} (Total in slice: {len(sliced_dataset)})")

    # 3. Path Validation with Extension Fallback (.jpg <-> .png)
    valid_dataset = []
    for data in sliced_dataset:
        image_path = os.path.join(images_dir, data["filename"])
        
        # Extension fallback check if path is missing
        if not os.path.exists(image_path):
            base_path, ext = os.path.splitext(image_path)
            alt_ext = ".png" if ext.lower() in [".jpg", ".jpeg"] else ".jpg"
            alt_path = base_path + alt_ext
            
            if os.path.exists(alt_path):
                image_path = alt_path
            else:
                print(f"Warning: Image not found at {image_path} (or {alt_path}), skipping.")
                continue

        valid_dataset.append({
            "id": data["id"],
            "image_path": image_path,
            "prompt": data["prompt"]
        })
        
    return valid_dataset

def main(args):
    torch.manual_seed(args.seed)

    # hugging face login
    if args.hf_token_path and os.path.exists(args.hf_token_path):
        with open(args.hf_token_path, "r") as f:
            login(token=f.read().strip())
        print("Logged in to HuggingFace using token file.")
    elif "HF_TOKEN" in os.environ:
        login(token=os.environ["HF_TOKEN"])
        print("Logged in to HuggingFace using HF_TOKEN environment variable.")
    else:
        print("No HF token file found; assuming public access or existing 'huggingface-cli login'.")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dry_run:
        print("[DRY RUN MODE] Skipping model loading. Using Mock Pipeline.")
        pipe = MockPipeline()
    else:
        print("[REAL RUN MODE] Loading models and pipeline...")
        MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
        
        print("Loading Text Encoders...")
        clip_text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16).to("cuda")
        t5_text_encoder = T5EncoderModel.from_pretrained(
            MODEL_ID, 
            subfolder="text_encoder_2", 
            torch_dtype=torch.bfloat16
        ).to("cuda")
        # quantizate after initialization
        # t5_text_encoder.to(dtype=torch.float8_e4m3fn, device="cuda")

        print("Loading VAE & Transformer...")
        vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16).to("cuda")
        transformer = FluxTransformer2DModel.from_pretrained(
            MODEL_ID, 
            subfolder="transformer",
            torch_dtype=torch.bfloat16
            #torch_dtype=torch.float8_e4m3fn
        ).to("cuda")

        print("Assembling Pipeline...")
        pipe = SliderEditFluxKontextPipeline.from_pretrained(
            MODEL_ID, vae=vae, text_encoder=clip_text_encoder, text_encoder_2=t5_text_encoder,
            transformer=transformer, torch_dtype=torch.bfloat16
        )

        pipe.load_gstlora(args.lora_path)
        pipe.loaded_adapter = LoRAAdapterType.GSTLORA
        pipe.set_progress_bar_config(disable=True) 

    # Load Dataset Slice
    dataset = load_dataset(args.mapping_file, args.images_dir, args.dataset_type, start_idx=args.start_idx, end_idx=args.end_idx)
    alpha_values = [1.0, 0.5, 0.0, -0.5, -1.0]

    print(f"Starting generation for {len(dataset)} images...")
    torch.backends.cudnn.benchmark = True 

    for data in tqdm(dataset, desc=f"GPU Process [{args.start_idx}:{args.end_idx}]"):
        image_output_dir = os.path.join(args.output_dir, data["id"])
        os.makedirs(image_output_dir, exist_ok=True)
        
        try:
            img = Image.open(data["image_path"]).convert("RGB")
        except Exception as e:
            print(f"Skipping {data['id']}: Could not load image. {e}")
            continue
            
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for step_idx, alpha in enumerate(alpha_values):
                save_path = os.path.join(image_output_dir, f"step_{step_idx:02d}.png")
                
                if os.path.exists(save_path):
                    continue
                    
                generator = torch.Generator("cuda").manual_seed(args.seed)
                output_image = pipe(
                    image=img, prompt=data["prompt"], generator=generator, slider_alpha=alpha
                ).images[0]
                
                output_image.save(save_path, format="PNG")

        torch.cuda.empty_cache()

    print(f"Completed slice [{args.start_idx}:{args.end_idx}] -> Saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SliderEdit across datasets.")
    
    # Dataset Arguments
    parser.add_argument("--dataset_type", type=str, choices=["pie-bench", "rs-objects"], required=True, help="Which dataset logic to use")
    parser.add_argument("--mapping_file", type=str, required=True, help="Path to mapping_file.json")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to source images folder")
    parser.add_argument("--start_idx", type=int, default=0, help="Start image index")
    parser.add_argument("--end_idx", type=int, default=None, help="End image index (exclusive)")
    
    # Configuration Arguments
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed")
    parser.add_argument("--output_dir", type=str, default="/tmp/slideredit_outputs", help="Output directory")
    parser.add_argument("--lora_path", type=str, default="./checkpoints/example_training_gstlora_iter500.safetensors", help="Path to LoRA")
    parser.add_argument("--hf_token_path", type=str, default="./.hf_token", help="Path to HF token")
    parser.add_argument("--dry_run", action="store_true", help="Run full pipeline logic without loading models/using VRAM")

    args = parser.parse_args()
    main(args)
