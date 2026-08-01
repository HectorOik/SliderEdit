import os
import sys
import json
import argparse
from PIL import Image
import torch

def run(dataset_type, mapping_file, images_dir, output_dir, hf_token_path, start_idx, end_idx, lora_path):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.append(repo_root)

    # ---------------------------------------------------------------------------
    # 1. HuggingFace Authentication
    # ---------------------------------------------------------------------------
    from huggingface_hub import login
    token_path = os.path.expanduser(hf_token_path)
    if os.path.exists(token_path):
        with open(token_path, "r") as f:
            login(token=f.read().strip())
        print("Logged in via cluster token file.")
    elif "HF_TOKEN" in os.environ:
        login(token=os.environ["HF_TOKEN"])
        print("Logged in via HF_TOKEN environment variable.")

    # ---------------------------------------------------------------------------
    # 2. Load SliderEdit Pipeline
    # ---------------------------------------------------------------------------
    try:
        from slideredit.pipelines import SliderEditFluxKontextPipeline
    except ImportError as e:
        print(f"❌ Failed to import SliderEdit pipeline: {e}")
        sys.exit(1)

    print("Loading SliderEdit model pipeline...")
    pipe = SliderEditFluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16
    ).to("cuda")

    if lora_path:
        print(f"Loading LoRA/Adapter weights from: {lora_path}")
        if os.path.isfile(lora_path):
            lora_dir = os.path.dirname(lora_path)
            weight_file = os.path.basename(lora_path)
        else:
            lora_dir = lora_path
            weight_file = "pytorch_lora_weights.safetensors"

        # Bypass pipeline wrapper bug and load directly using diffusers native method
        if "gst" in lora_path.lower():
            pipe.load_lora_weights(lora_dir, weight_name=weight_file, adapter_name="gstlora")
            pipe.set_adapters(["gstlora"])
        else:
            pipe.load_lora_weights(lora_dir, weight_name=weight_file, adapter_name="stlora")
            pipe.set_adapters(["stlora"])

    # Slider values specified: 1, 0.5, 0, -0.5, -1
    slider_alphas = [1.0, 0.5, 0.0, -0.5, -1.0]
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------------------------
    # 3. Load Dataset Mapping File (AnyText or PIE-bench)
    # ---------------------------------------------------------------------------
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"Cannot find mapping file at '{mapping_file}'")

    print(f"Loading dataset mapping from {mapping_file}...")
    with open(mapping_file, 'r') as f:
        dataset_records = json.load(f)

    if isinstance(dataset_records, dict):
        dataset_records = [{"id": k, **v} for k, v in dataset_records.items()]

    actual_end_idx = end_idx if end_idx is not None else len(dataset_records)
    dataset_slice = dataset_records[start_idx:actual_end_idx]
    print(f"Processing slice [{start_idx}:{actual_end_idx}] out of {len(dataset_records)} total items.")

    # ---------------------------------------------------------------------------
    # 4. Benchmark Execution Loop
    # ---------------------------------------------------------------------------
    for idx, row in enumerate(dataset_slice):
        current_idx = start_idx + idx
        
        if dataset_type == "anytext":
            sweep_id = str(row.get('image', row.get('id', f"{current_idx:012d}.jpg")))
            edit_prompt = str(row.get('prompt', row.get('target_prompt', row.get('caption', '')))).strip()
            subprompts = [edit_prompt]
        else:
            sweep_id = str(row.get('id', f"{current_idx:012d}.jpg"))
            edit_prompt = str(row.get('target_prompt', row.get('editing_instruction', ''))).strip()
            subprompts = [edit_prompt]

        img_filename = os.path.basename(sweep_id)
        seed = int(row.get('seed', 42))
        source_img_path = os.path.join(images_dir, img_filename)
        exp_dir = os.path.join(output_dir, os.path.splitext(img_filename)[0])
        os.makedirs(exp_dir, exist_ok=True)

        if not os.path.exists(source_img_path):
            print(f"⚠️ [{current_idx+1}/{actual_end_idx}] Skipping '{sweep_id}' (Missing image at {source_img_path})")
            continue

        print(f"\n[{current_idx+1}/{actual_end_idx}] Processing '{sweep_id}' with SliderEdit...")

        try:
            input_image = Image.open(source_img_path).convert("RGB")

            for alpha in slider_alphas:
                alpha_str = f"{alpha}".replace(".", "_").replace("-", "neg_")
                save_path = os.path.join(exp_dir, f"alpha_{alpha_str}.png")

                if "gst" not in lora_path.lower() and len(subprompts) > 1:
                    output = pipe(
                        image=input_image,
                        prompt=" and ".join(subprompts),
                        generator=torch.Generator("cuda").manual_seed(seed),
                        subprompts_list=subprompts,
                        slider_alpha_list=[alpha] * len(subprompts),
                    )
                else:
                    output = pipe(
                        image=input_image,
                        prompt=edit_prompt,
                        generator=torch.Generator("cuda").manual_seed(seed),
                        slider_alpha=alpha,
                    )

                out_img = output.images[0]
                if out_img.mode != "RGB":
                    out_img = out_img.convert("RGB")
                
                out_img.save(save_path)
                print(f"  -> Saved: {save_path}")

        except Exception as e:
            print(f"❌ Execution error for {sweep_id}: {e}")

    print("\n SliderEdit Benchmark Complete!")


def main():
    parser = argparse.ArgumentParser(description="Run SliderEdit Benchmarks")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["pie-bench", "anytext"], help="Dataset type")
    parser.add_argument("--mapping_file", type=str, required=True, help="Path to JSON mapping file")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to images directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save outputs")
    parser.add_argument("--hf_token_path", type=str, default="~/.hf_token", help="HuggingFace token path")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index")
    parser.add_argument("--end_idx", type=int, default=None, help="End index")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to STLoRA or GSTLoRA weights")

    args = parser.parse_args()

    run(
        dataset_type=args.dataset_type,
        mapping_file=args.mapping_file,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        hf_token_path=args.hf_token_path,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        lora_path=args.lora_path
    )

if __name__ == "__main__":
    main()
