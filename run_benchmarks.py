import torch
import os
import json
import glob
import argparse
from PIL import Image
from tqdm import tqdm
from huggingface_hub import login
from transformers import CLIPTextModel, T5EncoderModel, BitsAndBytesConfig
from diffusers import AutoencoderKL
from diffusers.models import FluxTransformer2DModel
from slideredit.pipelines import SliderEditFluxKontextPipeline, LoRAAdapterType

# for debugging purposes
class MockPipeline:
    def __call__(self, image, prompt, generator, slider_alpha):
        class DummyOutput:
            def __init__(self, img):
                self.images = [img.copy()]
        return DummyOutput(image)

def load_dataset(mapping_file_path_or_dir, images_dir, dataset_type, start_idx=0, end_idx=None):
    """
    Parses dataset sources (Parquet files with embedded/referenced images for PIE-bench or JSON files for rs-objects) and slices the workload.
    """
    dataset = []
    
    # 1. Parsing Logic
    if dataset_type == "pie-bench":
        print(f"Loading PIE-bench from parquet files under {mapping_file_path_or_dir}...")
        import pandas as pd
        
        if os.path.isdir(mapping_file_path_or_dir):
            parquet_files = glob.glob(os.path.join(mapping_file_path_or_dir, "**/*.parquet"), recursive=True)
        else:
            parquet_files = [mapping_file_path_or_dir]
            
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found for PIE-bench at {mapping_file_path_or_dir}")
            
        for p_file in parquet_files:
            df = pd.read_parquet(p_file)
            for idx, row in df.iterrows():
                sample_id = str(row.get("id", f"sample_{idx}"))
                target_prompt = str(row.get("target_prompt", ""))
                source_prompt = str(row.get("source_prompt", ""))
                
                # Extract embedded image bytes from parquet if available
                img_obj = row.get("image", None)
                img_path = None
                
                if isinstance(img_obj, dict) and "bytes" in img_obj:
                    img_bytes = img_obj["bytes"]
                    temp_img_dir = os.path.join(images_dir, "_extracted_cache")
                    os.makedirs(temp_img_dir, exist_ok=True)
                    img_path = os.path.join(temp_img_dir, f"{sample_id}.jpg")
                    if not os.path.exists(img_path):
                        with open(img_path, "wb") as f_img:
                            f_img.write(img_bytes)
                elif isinstance(img_obj, bytes):
                    temp_img_dir = os.path.join(images_dir, "_extracted_cache")
                    os.makedirs(temp_img_dir, exist_ok=True)
                    img_path = os.path.join(temp_img_dir, f"{sample_id}.jpg")
                    if not os.path.exists(img_path):
                        with open(img_path, "wb") as f_img:
                            f_img.write(img_obj)
                else:
                    img_filename = str(row.get("path", f"{sample_id}.jpg"))
                    img_path = os.path.join(images_dir, img_filename)

                dataset.append({
                    "id": sample_id, 
                    "image_path": img_path, 
                    "prompt": target_prompt,
                    "source_prompt": source_prompt
                })
                
    elif dataset_type == "rs-objects":
        print(f"Loading rs-objects mapping from {mapping_file_path_or_dir}...")
        with open(mapping_file_path_or_dir, 'r') as f:
            mapping_data = json.load(f)
            
        if isinstance(mapping_data, list):
            for item in mapping_data:
                img_filename = item.get("image", item.get("image_id", ""))
                prompt = item.get("text", item.get("prompt", ""))
                clean_id = os.path.splitext(os.path.basename(img_filename))[0]
                image_path = os.path.join(images_dir, img_filename)
                dataset.append({"id": clean_id, "image_path": image_path, "prompt": prompt, "source_prompt": ""})
        elif isinstance(mapping_data, dict):
            for img_filename, prompt in mapping_data.items():
                clean_id = os.path.splitext(os.path.basename(img_filename))[0]
                image_path = os.path.join(images_dir, img_filename)
                dataset.append({"id": clean_id, "image_path": image_path, "prompt": prompt, "source_prompt": ""})

    # 2. Slicing Logic for Multi-GPU
    sliced_dataset = dataset[start_idx:end_idx] if end_idx is not None else dataset[start_idx:]
    print(f"Processing slice: indices {start_idx} to {start_idx + len(sliced_dataset)} (Total in slice: {len(sliced_dataset)})")

    # 3. Path Validation with Extension Fallback (.jpg <-> .png)
    valid_dataset = []
    for data in sliced_dataset:
        image_path = data["image_path"]
        
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
            "prompt": data["prompt"],
            "source_prompt": data["source_prompt"]
        })
        
    return valid_dataset

def load_dataset_stratified_pie_bench(mapping_file_path_or_dir, images_dir, samples_per_category=20):
    """
    Loads PIE-bench and ensures a strict stratified split 
    (e.g., first N samples from each of the 10 category subfolders/files).
    """
    import pandas as pd
    valid_dataset = []
    
    if os.path.isdir(mapping_file_path_or_dir):
        # Find category directories or parquet files grouped by category
        category_dirs = sorted([os.path.join(mapping_file_path_or_dir, d) for d in os.listdir(mapping_file_path_or_dir) if os.path.isdir(os.path.join(mapping_file_path_or_dir, d))])
        
        # Fallback if parquet files are flat in a single folder
        if not category_dirs:
            category_dirs = [mapping_file_path_or_dir]
            
        for cat_dir in category_dirs:
            cat_name = os.path.basename(cat_dir)
            if cat_name.startswith('.'):  # Skip hidden dirs like .cache
                continue
            parquet_files = glob.glob(os.path.join(cat_dir, "*.parquet"))
            
            cat_samples_collected = 0
            for p_file in sorted(parquet_files):
                df = pd.read_parquet(p_file)
                for _, row in df.iterrows():
                    if cat_samples_collected >= samples_per_category:
                        break
                        
                    sample_id = str(row.get("id", ""))
                    img_filename = str(row.get("path", f"{sample_id}.jpg"))
                    target_prompt = str(row.get("target_prompt", ""))
                    source_prompt = str(row.get("source_prompt", ""))
                    
                    # Resolve image path with extension fallback
                    image_path = os.path.join(images_dir, img_filename)
                    if not os.path.exists(image_path):
                        base_path, ext = os.path.splitext(image_path)
                        alt_ext = ".png" if ext.lower() in [".jpg", ".jpeg"] else ".jpg"
                        image_path = base_path + alt_ext
                        
                    if os.path.exists(image_path):
                        valid_dataset.append({
                            "id": sample_id,
                            "image_path": image_path,
                            "prompt": target_prompt,
                            "source_prompt": source_prompt,
                            "category": cat_name
                        })
                        cat_samples_collected += 1
                        
                if cat_samples_collected >= samples_per_category:
                    break
            print(f"Category [{cat_name}]: Loaded {cat_samples_collected} samples.")
            
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

        # ============= Quantization Logic Start ======================
        
        print("Loading Text Encoders...")
        clip_text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16).to("cuda")
        t5_text_encoder = T5EncoderModel.from_pretrained(
            MODEL_ID, 
            subfolder="text_encoder_2", 
            torch_dtype=torch.bfloat16
        ).to("cuda")

        print("Loading VAE...")
        vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16).to("cuda")

        # 4-bit quantization config for the massive Flux Transformer (~12B params)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        print("⚡ Loading Quantized FLUX Transformer (4-bit NF4)...")
        transformer = FluxTransformer2DModel.from_pretrained(
            MODEL_ID, 
            subfolder="transformer",
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda"}
        )

        print("Assembling Pipeline...")
        pipe = SliderEditFluxKontextPipeline.from_pretrained(
            MODEL_ID, vae=vae, text_encoder=clip_text_encoder, text_encoder_2=t5_text_encoder,
            transformer=transformer, torch_dtype=torch.bfloat16
        )

        pipe.load_gstlora(args.lora_path)
        pipe.loaded_adapter = LoRAAdapterType.GSTLORA
        pipe.set_progress_bar_config(disable=True) 

    # Load Dataset Slice
    # dataset = load_dataset(args.mapping_file, args.images_dir, args.dataset_type, start_idx=args.start_idx, end_idx=args.end_idx)
    dataset = load_dataset_stratified_pie_bench(args.mapping_file, args.images_dir, samples_per_category=20)
    print(f"Total stratified dataset size loaded: {len(dataset)} images.")
    alpha_values = [1.0, 0.5, 0.0, -0.5, -1.0]

    print(f"Starting generation for {len(dataset)} images...")
    torch.backends.cudnn.benchmark = True 

    for data in tqdm(dataset, desc=f"GPU Process [{args.start_idx}:{args.end_idx}]"):
        image_output_dir = os.path.join(args.output_dir, data["id"])
        os.makedirs(image_output_dir, exist_ok=True)
        
        try:
            img = Image.open(data["image_path"]).convert("RGB")
            img = img.resize((512, 512), Image.Resampling.LANCZOS)
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
    parser.add_argument("--mapping_file", type=str, required=True, help="Path to mapping file or PIE-bench parquet folder directory")
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
