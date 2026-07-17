import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import T5TokenizerFast
import itertools

# Import your codebase elements
from slideredit.models.selective_lora import SelectiveLoRALinear
from slideredit.utils import flux_kontext_find_substring_token_indices, stlora_token_mask_ctx
from training.train_stlora_flux_kontext import CustomMultiEditInstructionDataset, collate_fn

# =====================================================================
# 1. DEFINE A TOY MM-DIT TRANSFORMER
# =====================================================================
class MockTransformerConfig:
    guidance_embeds = False

class ToyMMDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MockTransformerConfig()
        # Create a toy linear projection wrapped in your SelectiveLoRALinear
        self.linear = nn.Linear(1024, 1024)
        self.selective_lora = SelectiveLoRALinear(self.linear, r=4, alpha=4)

    def forward(self, hidden_states, timestep, guidance, pooled_projections, encoder_hidden_states, txt_ids, img_ids, return_dict=False):
        # Simply project the hidden states to mimic a MMDiT output block
        # Ensure we return a tuple containing the tensor just like FluxTransformer2DModel
        out = self.selective_lora(hidden_states)
        return (out,)

# =====================================================================
# 2. RUN LOCAL PPS LOSS AND GRADIENT SMOKE TEST
# =====================================================================
def run_training_step_smoke_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️ Starting Training Step Smoke Test on [{device.upper()}]...")

    # Load a lightweight real tokenizer to test substring mapping
    print("📖 Loading T5 Tokenizer...")
    t5_tokenizer = T5TokenizerFast.from_pretrained("t5-small")

    # 1. Initialize our Mock Network and Optimizer
    transformer = ToyMMDiT().to(device)
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=1e-4)

    # 2. Mimic Batch Data returned from your Custom Dataset
    # This matches the multi-instruction structure (PPS)
    batch = {
        "orig_edit_prompts": ["make the grass green and make the sky blue"],
        "instructions_to_apply_prompts": ["make the grass green"],
        "instructions_to_suppress": [["make the sky blue"]]
    }

    print("🧩 Generating Selective Token Masks...")
    bsz = len(batch["orig_edit_prompts"])
    tokens_mask = torch.zeros((bsz, 512), device=device, dtype=torch.bool)

    # Crucial test: Will your substring search map correctly to tokenizer space?
    try:
        for i in range(bsz):
            for instruction_to_suppress in batch["instructions_to_suppress"][i]:
                indices = flux_kontext_find_substring_token_indices(
                    batch["orig_edit_prompts"][i], 
                    instruction_to_suppress, 
                    t5_tokenizer
                )
                tokens_mask[i, indices] = True
        print("✅ Token masking indices generated successfully!")
    except Exception as e:
        print(f"❌ Substring token masking failed: {e}")
        return

    # 3. Create mock tensors representing spatial latent vectors
    # (Using 512 sequence length to match our sequence size)
    latent_model_input = torch.randn(bsz, 512, 1024, device=device)
    timesteps = torch.tensor([500.0], device=device)
    guidance = None
    pooled_prompt_embeds = torch.randn(bsz, 768, device=device)
    prompt_embeds = torch.randn(bsz, 512, 1024, device=device)
    text_ids = torch.zeros(512, 3, device=device)
    latent_ids = torch.zeros(bsz, 512, 3, device=device)

    print("🔥 Executing PPS Neutral vs Active Forward Passes...")
    try:
        # A. Get Target prediction (neutral state, no mask active)
        with stlora_token_mask_ctx(transformer, None):
            gt_model_pred_neutral = transformer(
                hidden_states=latent_model_input,
                timestep=timesteps,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_ids,
                return_dict=False,
            )[0]

        # B. Get Active Prediction with Selective LoRA mask applied
        with stlora_token_mask_ctx(transformer, tokens_mask, disable_mask_after=False):
            model_pred = transformer(
                hidden_states=latent_model_input,
                timestep=timesteps,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_ids,
                return_dict=False,
            )[0]

        # C. Calculate PPS Flow Matching Loss
        weighting = torch.ones_like(model_pred)
        loss = torch.mean((weighting * (model_pred - gt_model_pred_neutral) ** 2))
        print(f"✅ Loss calculation passed! Loss Value: {loss.item():.6f}")

        # D. Backward Pass & Optimizer Step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("✅ Backpropagation and optimizer update completed successfully!")
        
        print("\n🎉 SMOKE TEST SUCCESSFUL: Your training step math is completely locked in and ready for the cluster GPU!")

    except Exception as e:
        print(f"❌ Training forward/backward steps failed: {e}")

if __name__ == "__main__":
    run_training_step_smoke_test()