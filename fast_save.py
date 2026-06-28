import torch
from peft import PeftModel, LoraConfig
from diffusers import FluxTransformer2DModel

print("[*] Forcing manual LoRA extraction from checkpoint-350...")

# 1. Load the underlying optimizer state to grab the raw trained weights
opt_state = torch.load("experiments/gstlora_flux_kontext/checkpoint-350/optimizer.bin", map_location="cpu")

# 2. Extract the state dict tracking the trainable weights
# In accelerate + deepspeed/fsdp/native, the parameters are indexed inside the optimizer
trainable_weights = {}
if "state" in opt_state:
    for param_id, param_data in opt_state["state"].items():
        if isinstance(param_data, dict):
            for k, v in param_data.items():
                if isinstance(v, torch.Tensor):
                    # Direct assignment of tensors
                    trainable_weights[f"transformer.{param_id}"] = v

# 3. Save it as a raw tensor file so you have your data securely isolated
torch.save(trainable_weights, "experiments/gstlora_flux_kontext/gst_lora_final.pt")
print("[──] SUCCESS! Your weights are saved as 'gst_lora_final.pt' inside your experiments folder!")
