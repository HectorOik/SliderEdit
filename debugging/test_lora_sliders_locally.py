import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import os

from slideredit.models.selective_lora import SelectiveLoRALinear

def run_local_smoke_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Booting Local Tensor Sandbox on device: [{device.upper()}]")

    base_layer = nn.Linear(1024, 1024, bias=False)

    # initialize with random weights
    nn.init.normal_(base_layer.weight)

    print("Attaching SelectiveLoRALinear to base layer...")
    lora_layer = SelectiveLoRALinear(base_layer, r=4, alpha=1.0)

    print("🧠 Dynamically waking up internal LoRA parameters...")
    found_any = False
    for name, param in lora_layer.named_parameters():
        # We target parameters that aren't the base layer weight
        if "weight" in name and "base_layer" not in name:
            print(f"  -> Initializing param: {name}")
            nn.init.normal_(param, std=0.1)
            found_any = True
            
    if not found_any:
        print("  ⚠️ Warning: No explicit child weights detected. Checking submodules...")
        for name, buf in lora_layer.named_buffers():
            print(f"  -> Found buffer: {name}")

    # create dummy "image latent" variable
    # Shape: (Batch=1, Sequence=4096, Features=1024)
    seq_len = 512 # replicating actual architecture
    dummy_input = torch.randn(1, seq_len, 1024).to(device)
    
    dummy_mask = torch.ones(1, seq_len, dtype=torch.bool).to(device)
    dummy_mask[0, 10:15] = True
    dummy_mask[0, 20:25] = True # set random token indices to True for testing

    lora_layer.set_token_mask(dummy_mask)

    print("RUNNING SLIDER SCALE TEST...")

    # Test 1: Default scaling
    test_scale = [-1.0, 0.0, 1.0]

    results_cache = {}

    for lora_scale_0, lora_scale_1 in itertools.product(test_scale, repeat=2):
        # Build a proper, broadcastable Scale Tensor of shape (Batch=1, Sequence=512, Features=1)
        # Using the same dtype as our dummy input to prevent cast conflicts
        scaling_tensor = torch.ones(1, seq_len, 1, dtype=dummy_input.dtype).to(device)
        
        # Apply the scale factor across the token sequence
        scaling_tensor[0, 10:15, 0] = lora_scale_0  # First slider scale
        scaling_tensor[0, 20:25, 0] = lora_scale_1  # Second slider scale

        # Pass this scale tensor to the lora layer
        lora_layer.set_scaling(scaling_tensor)

        # forward pass
        with torch.no_grad():
            output = lora_layer(dummy_input)

        # calculate sum of output tensor to get a single numeric "fingerprint" for comparison
        tensor_fingerprint = output.sum().item()

        # store results for comparison
        results_cache[f"Scale [{lora_scale_0}, {lora_scale_1}]"] = tensor_fingerprint
        print(f"Slider [{lora_scale_0:>4}, {lora_scale_1:>4}] -> Output Fingerprint: {tensor_fingerprint:.4f}")

    # Analyze Results
    print("\n📊 ANALYSIS:")
    unique_fingerprints = set(results_cache.values())
    if len(unique_fingerprints) == 1:
        print("❌ FAIL: All outputs are absolutely identical. The sliders are completely dead.")
        print("Diagnosis: The set_scaling() method is not physically altering the forward pass math.")
    else:
        print("✅ SUCCESS: The outputs changed! The sliders are physically altering the math.")
        print("Diagnosis: The LoRA layer works perfectly. The bug is inside the Pipeline's generation loop.")


if __name__ == "__main__":
    run_local_smoke_test()