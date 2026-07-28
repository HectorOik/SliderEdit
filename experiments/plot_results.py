import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SLIDEREDIT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
THESIS_DIR = os.path.abspath(os.path.join(SLIDEREDIT_DIR, ".."))

def parse_args():
    parser = argparse.ArgumentParser(description="Plot SliderEdit evaluation results from CSV.")
    parser.add_argument(
        "--csv_path",
        type=str,
        default=os.path.join(THESIS_DIR, "outputs", "slideredit", "stlora", "evaluation_results.csv"),
        help="Path to evaluation_results.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(THESIS_DIR, "outputs", "slideredit", "stlora", "plots"),
        help="Directory to save generated plot PNGs",
    )
    return parser.parse_args()


def setup_plot_style():
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'figure.titlesize': 15,
    })


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_plot_style()

    print("=" * 60)
    print("SliderEdit Results Plotting Pipeline (Corrected Scale Convention)")
    print("Convention: 1.0 = Preserved Base, 0.0 = Full Edit, -1.0 = Strong Edit")
    print("=" * 60)

    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"Evaluation CSV not found at {args.csv_path}")

    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} records for {df['id'].nunique()} unique IDs across {df['domain'].nunique()} domains.")

    domains = df['domain'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(domains)))

    # X-axis scale order: 1.0 (Preserved) -> 0.0 (Full Edit) -> -1.0 (Strong Edit)
    scale_order = [1.0, 0.0, -1.0]

    # -------------------------------------------------------------
    # 1. VLM Alignment vs. Slider 1 Scale (Preserved 1.0 -> Full 0.0 -> Strong -1.0)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics_vlm = [
        ("VLM_CLIP_subprompt1", "CLIP Score (Subprompt 1)"),
        ("VLM_Siglip_subprompt1", "SigLIP Score (Subprompt 1)"),
        ("VLM_BLIP_subprompt1", "BLIP ITM Score (Subprompt 1)"),
    ]

    for i, (col_name, title) in enumerate(metrics_vlm):
        ax = axes[i]
        if col_name in df.columns:
            grouped = df.groupby(['domain', 'slider1_val'])[col_name].mean().reset_index()
            for j, dom in enumerate(domains):
                dom_data = grouped[grouped['domain'] == dom].sort_values(by='slider1_val', ascending=False)
                ax.plot(dom_data['slider1_val'], dom_data[col_name], marker='o', linewidth=2, label=dom, color=colors[j])
            
            overall = df.groupby('slider1_val')[col_name].mean().reset_index().sort_values(by='slider1_val', ascending=False)
            ax.plot(overall['slider1_val'], overall[col_name], marker='s', linewidth=3, linestyle='--', label='Overall Mean', color='black')

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Slider Scale [ 1.0: Preserved  ->  0.0: Full Edit  ->  -1.0: Strong Edit ]")
        ax.set_ylabel("VLM Score (Higher = Stronger Match)")
        ax.set_xlim(1.2, -1.2)  # Invert x-axis to show Preserved -> Edit progression
        ax.legend(frameon=True)

    fig.suptitle("VLM Alignment vs. Edit Scale (Subprompt 1)", fontsize=15, y=1.03)
    fig.tight_layout()
    plot1_path = os.path.join(args.output_dir, "vlm_scores_slider1.png")
    fig.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {plot1_path}")

    # -------------------------------------------------------------
    # 2. VLM Alignment vs. Slider 2 Scale
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics_vlm2 = [
        ("VLM_CLIP_subprompt2", "CLIP Score (Subprompt 2)"),
        ("VLM_Siglip_subprompt2", "SigLIP Score (Subprompt 2)"),
        ("VLM_BLIP_subprompt2", "BLIP ITM Score (Subprompt 2)"),
    ]

    for i, (col_name, title) in enumerate(metrics_vlm2):
        ax = axes[i]
        if col_name in df.columns:
            grouped = df.groupby(['domain', 'slider2_val'])[col_name].mean().reset_index()
            for j, dom in enumerate(domains):
                dom_data = grouped[grouped['domain'] == dom].sort_values(by='slider2_val', ascending=False)
                ax.plot(dom_data['slider2_val'], dom_data[col_name], marker='o', linewidth=2, label=dom, color=colors[j])
            
            overall = df.groupby('slider2_val')[col_name].mean().reset_index().sort_values(by='slider2_val', ascending=False)
            ax.plot(overall['slider2_val'], overall[col_name], marker='s', linewidth=3, linestyle='--', label='Overall Mean', color='black')

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Slider Scale [ 1.0: Preserved  ->  0.0: Full Edit  ->  -1.0: Strong Edit ]")
        ax.set_ylabel("VLM Score (Higher = Stronger Match)")
        ax.set_xlim(1.2, -1.2)
        ax.legend(frameon=True)

    fig.suptitle("VLM Alignment vs. Edit Scale (Subprompt 2)", fontsize=15, y=1.03)
    fig.tight_layout()
    plot2_path = os.path.join(args.output_dir, "vlm_scores_slider2.png")
    fig.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {plot2_path}")

    # -------------------------------------------------------------
    # 3. Feature Distance Metrics vs. Slider 1 Scale
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    metrics_dist = [
        ("Dist_LPIPS_vgg", "LPIPS (VGG) Distance"),
        ("Dist_LPIPS_alex", "LPIPS (AlexNet) Distance"),
        ("Dist_DiNOv2", "DINOv2 Feature Distance"),
        ("Dist_FaceID", "FaceID Distance"),
    ]

    for i, (col_name, title) in enumerate(metrics_dist):
        ax = axes[i]
        if col_name in df.columns and not df[col_name].isna().all():
            grouped = df.groupby(['domain', 'slider1_val'])[col_name].mean().reset_index()
            for j, dom in enumerate(domains):
                dom_data = grouped[grouped['domain'] == dom].sort_values(by='slider1_val', ascending=False)
                if not dom_data[col_name].isna().all():
                    ax.plot(dom_data['slider1_val'], dom_data[col_name], marker='o', linewidth=2, label=dom, color=colors[j])
            
            overall = df.groupby('slider1_val')[col_name].mean().reset_index().sort_values(by='slider1_val', ascending=False)
            ax.plot(overall['slider1_val'], overall[col_name], marker='s', linewidth=3, linestyle='--', label='Overall Mean', color='black')

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Slider Scale [ 1.0: Preserved  ->  0.0: Full Edit  ->  -1.0: Strong Edit ]")
        ax.set_ylabel("Distance from Reference")
        ax.set_xlim(1.2, -1.2)
        ax.legend(frameon=True)

    fig.suptitle("Distance from Preserved Image vs. Edit Scale", fontsize=15, y=1.03)
    fig.tight_layout()
    plot3_path = os.path.join(args.output_dir, "feature_distances_slider1.png")
    fig.savefig(plot3_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {plot3_path}")

    # -------------------------------------------------------------
    # 4. Domain Bar Comparison (CLIP vs LPIPS vs DINOv2)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    domain_means = df.groupby('domain')[['VLM_CLIP_subprompt1', 'Dist_LPIPS_vgg', 'Dist_DiNOv2']].mean().reset_index()

    sns.barplot(data=domain_means, x='domain', y='VLM_CLIP_subprompt1', hue='domain', ax=axes[0], palette='crest', legend=False)
    axes[0].set_title("Mean CLIP Score (Subprompt 1)", fontweight='bold')
    axes[0].set_xlabel("")
    axes[0].tick_params(axis='x', rotation=30)

    sns.barplot(data=domain_means, x='domain', y='Dist_LPIPS_vgg', hue='domain', ax=axes[1], palette='flare', legend=False)
    axes[1].set_title("Mean LPIPS (VGG) Distance", fontweight='bold')
    axes[1].set_xlabel("")
    axes[1].tick_params(axis='x', rotation=30)

    sns.barplot(data=domain_means, x='domain', y='Dist_DiNOv2', hue='domain', ax=axes[2], palette='viridis', legend=False)
    axes[2].set_title("Mean DINOv2 Distance", fontweight='bold')
    axes[2].set_xlabel("")
    axes[2].tick_params(axis='x', rotation=30)

    fig.suptitle("Mean Evaluation Metrics Across Domains", fontsize=15, y=1.03)
    fig.tight_layout()
    plot4_path = os.path.join(args.output_dir, "domain_comparison_barchart.png")
    fig.savefig(plot4_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {plot4_path}")

    print("=" * 60)
    print(f"SUCCESS: Generated 4 updated plots in: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
