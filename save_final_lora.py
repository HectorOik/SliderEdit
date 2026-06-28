import torch
from peft import get_peft_model_state_dict
from diffusers import FluxPipeline

output_dir = "./experiments/gstlora_flux_kontext/checkpoint-350"

print("[*] Re-packaging adapter layers from step 350 configuration...")

opt_state = torch.load(f"{output_dir}/optimizer.bin", map_location="cpu")

model_state_dict={}
if "state" in opt_state:
    for param_id, param_data in opt_state["state"].items():
        if isinstance(param_data, dict):
            pass

FluxPipeline.save_lora_weights(
    save_directory="./experiments/gstlora_flux_kontext/",
    transformer_lora_layers=opt_state,
    safe_serialization=True
)

print("[-] Success!")
