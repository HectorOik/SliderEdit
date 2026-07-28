import os
import sys
import re
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch

# Ensure SliderEdit directory is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SLIDEREDIT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
THESIS_DIR = os.path.abspath(os.path.join(SLIDEREDIT_DIR, ".."))

if SLIDEREDIT_DIR not in sys.path:
    sys.path.insert(0, SLIDEREDIT_DIR)

from evaluation.evaluators import (
    CLIPEvaluator,
    SiglipEvaluator,
    BLIPEvaluator,
    LPIPSFeatureDistanceEvaluator,
    DiNOFeatureDistanceEvaluator,
    FaceIdentityDistanceEvaluator,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation on generated SliderEdit images using VLM and Feature Distance metrics.")
    parser.add_argument(
        "--datasets_dir",
        type=str,
        default=os.path.join(THESIS_DIR, "datasets"),
        help="Path to datasets directory containing experiments24-7.csv and test_images",
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        default=os.path.join(THESIS_DIR, "outputs", "slideredit", "stlora"),
        help="Path to outputs directory containing generated image folders for each ID",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=os.path.join(THESIS_DIR, "outputs", "slideredit", "stlora", "evaluation_results.csv"),
        help="Path to save output CSV containing detailed evaluation metrics",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation models on ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--skip_faceid",
        action="store_true",
        help="Skip FaceIdentityDistanceEvaluator if set",
    )
    return parser.parse_args()


def parse_slider_values_from_filename(filename: str, id_name: str):
    """
    Extracts slider values (e.g. 0.0, 1.0, -1.0) from filenames like:
    portrait_02_subprompt1_0.00_subprompt2_neg_1.00.png
    """
    stem = os.path.splitext(filename)[0]
    if stem.startswith(id_name):
        stem = stem[len(id_name):]
    
    clean_stem = stem.replace("neg_", "-")
    matches = re.findall(r"[-+]?\d+\.\d+", clean_stem)
    if not matches:
        matches = re.findall(r"[-+]?\d+", clean_stem)
        
    floats = [float(m) for m in matches]
    
    val1 = floats[0] if len(floats) > 0 else None
    val2 = floats[1] if len(floats) > 1 else None
    
    return val1, val2


def main():
    args = parse_args()
    print("=" * 60)
    print("SliderEdit Image Evaluation Pipeline")
    print("=" * 60)
    print(f"Script Location:    {SCRIPT_DIR}")
    print(f"Datasets Directory: {args.datasets_dir}")
    print(f"Outputs Directory:  {args.outputs_dir}")
    print(f"Output CSV:         {args.output_csv}")
    print(f"Device:             {args.device}")
    print("=" * 60)

    # 1. Load Metadata
    metadata_csv_path = os.path.join(args.datasets_dir, "experiments24-7.csv")
    test_images_dir = os.path.join(args.datasets_dir, "test_images")

    if not os.path.exists(metadata_csv_path):
        raise FileNotFoundError(f"Metadata CSV not found at {metadata_csv_path}")

    df_meta = pd.read_csv(metadata_csv_path)
    meta_lookup = df_meta.set_index("id").to_dict(orient="index")
    print(f"Loaded metadata for {len(meta_lookup)} image IDs from {metadata_csv_path}")

    # 2. Check Outputs Directory
    if not os.path.exists(args.outputs_dir):
        raise FileNotFoundError(f"Outputs directory not found at {args.outputs_dir}")

    id_dirs = [d for d in os.listdir(args.outputs_dir) if os.path.isdir(os.path.join(args.outputs_dir, d))]
    print(f"Found {len(id_dirs)} generated image directories in {args.outputs_dir}")

    # 3. Initialize Evaluators
    print("\n--- Initializing Evaluators ---")
    vlm_evaluators = {}
    feature_evaluators = {}

    try:
        print("Loading CLIP Evaluator...")
        vlm_evaluators["CLIP"] = CLIPEvaluator(device=args.device)
    except Exception as e:
        print(f"Warning: Failed to load CLIP Evaluator: {e}")

    try:
        print("Loading SigLIP Evaluator...")
        vlm_evaluators["Siglip"] = SiglipEvaluator(device=args.device)
    except Exception as e:
        print(f"Warning: Failed to load SigLIP Evaluator: {e}")

    try:
        print("Loading BLIP Evaluator...")
        vlm_evaluators["BLIP"] = BLIPEvaluator(device=args.device)
    except Exception as e:
        print(f"Warning: Failed to load BLIP Evaluator: {e}")

    try:
        print("Loading LPIPS (VGG) Evaluator...")
        feature_evaluators["LPIPS_vgg"] = LPIPSFeatureDistanceEvaluator(net="vgg", device=args.device)
    except Exception as e:
        print(f"Warning: Failed to load LPIPS VGG Evaluator: {e}")

    try:
        print("Loading LPIPS (Alex) Evaluator...")
        feature_evaluators["LPIPS_alex"] = LPIPSFeatureDistanceEvaluator(net="alex", device=args.device)
    except Exception as e:
        print(f"Warning: Failed to load LPIPS Alex Evaluator: {e}")

    try:
        print("Loading DINOv2 Evaluator...")
        feature_evaluators["DiNOv2"] = DiNOFeatureDistanceEvaluator(device=args.device)
    except Exception as e:
        print(f"Warning: Failed to load DINOv2 Evaluator: {e}")

    if not args.skip_faceid:
        try:
            print("Loading FaceID Evaluator...")
            feature_evaluators["FaceID"] = FaceIdentityDistanceEvaluator()
        except Exception as e:
            print(f"Warning: Failed to load FaceID Evaluator: {e}")

    print("Evaluators loaded successfully!\n")

    # 4. Evaluation Loop
    results = []

    for id_name in tqdm(id_dirs, desc="Evaluating Image IDs"):
        if id_name not in meta_lookup:
            print(f"Warning: ID '{id_name}' not found in metadata CSV. Skipping.")
            continue

        id_info = meta_lookup[id_name]
        subprompt_1 = str(id_info.get("subprompt_1", ""))
        subprompt_2 = str(id_info.get("subprompt_2", ""))
        base_prompt = str(id_info.get("base_prompt", ""))
        domain = str(id_info.get("domain", ""))
        test_focus = str(id_info.get("test_focus", ""))

        # Locate reference image
        ref_path = os.path.join(test_images_dir, f"{id_name}.png")
        if not os.path.exists(ref_path):
            ref_path = os.path.join(test_images_dir, f"{id_name}.jpg")
            if not os.path.exists(ref_path):
                print(f"Warning: Reference image for '{id_name}' not found in {test_images_dir}. Skipping.")
                continue

        ref_image = Image.open(ref_path).convert("RGB")

        gen_dir_path = Path(os.path.join(args.outputs_dir, id_name)).resolve()
        gen_image_paths = sorted(list(gen_dir_path.glob("*.png")) + list(gen_dir_path.glob("*.jpg")))

        meta_json_path = os.path.join(gen_dir_path, "meta.json")
        folder_meta = {}
        if os.path.exists(meta_json_path):
            try:
                with open(meta_json_path, "r", encoding="utf-8") as f:
                    folder_meta = json.load(f)
            except Exception:
                pass

        for gen_path in gen_image_paths:
            filename = os.path.basename(gen_path)
            if filename == "meta.json":
                continue

            abs_path = str(gen_path)
            if os.name == 'nt' and not abs_path.startswith('\\\\?\\'):
                abs_path = '\\\\?\\' + abs_path

            try:
                gen_image = Image.open(abs_path).convert("RGB")
            except Exception as e:
                print(f"Error opening image {filename}: {e}")
                continue

            val1, val2 = parse_slider_values_from_filename(filename, id_name)

            record = {
                "id": id_name,
                "domain": domain,
                "file_name": filename,
                "slider1_val": val1,
                "slider2_val": val2,
                "subprompt_1": subprompt_1,
                "subprompt_2": subprompt_2,
                "base_prompt": base_prompt,
                "test_focus": test_focus,
            }

            # --- VLM Alignment Scores ---
            for name, evaluator in vlm_evaluators.items():
                if subprompt_1:
                    record[f"VLM_{name}_subprompt1"] = evaluator.get_score(gen_image, subprompt_1)
                if subprompt_2:
                    record[f"VLM_{name}_subprompt2"] = evaluator.get_score(gen_image, subprompt_2)
                if base_prompt:
                    record[f"VLM_{name}_base_prompt"] = evaluator.get_score(gen_image, base_prompt)

            # --- Feature & Identity Distance Metrics ---
            for name, evaluator in feature_evaluators.items():
                try:
                    dist = evaluator.get_distance(ref_image, gen_image)
                    record[f"Dist_{name}"] = dist if dist is not None else np.nan
                except Exception as e:
                    record[f"Dist_{name}"] = np.nan

            results.append(record)

    # 5. Save Results to CSV
    if not results:
        print("No evaluation results generated.")
        return

    df_results = pd.DataFrame(results)

    output_dir = os.path.dirname(os.path.abspath(args.output_csv))
    os.makedirs(output_dir, exist_ok=True)

    df_results.to_csv(args.output_csv, index=False)
    print("=" * 60)
    print(f"SUCCESS: Saved detailed evaluation results ({len(df_results)} rows) to:")
    print(f" -> {args.output_csv}")
    print("=" * 60)

    # 6. Display Summary Table
    numeric_cols = [c for c in df_results.columns if c.startswith("VLM_") or c.startswith("Dist_")]
    if numeric_cols:
        print("\n--- Summary Statistics (Mean Scores Across Dataset) ---")
        summary_df = df_results[numeric_cols].mean().to_frame(name="Mean")
        summary_df["Std"] = df_results[numeric_cols].std()
        print(summary_df.to_string())

        print("\n--- Mean Scores by Domain ---")
        domain_summary = df_results.groupby("domain")[numeric_cols].mean()
        print(domain_summary.to_string())


if __name__ == "__main__":
    main()
