import torch
import time
from huggingface_hub import login
from PIL import Image
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import BOTH configs under different names
from transformers import CLIPTextModel, T5EncoderModel
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from diffusers import AutoencoderKL
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers.models import FluxTransformer2DModel
from slideredit.pipelines import SliderEditFluxKontextPipeline, LoRAAdapterType

with open("/cs/student/msc/ml/2025/eoikonom/.hf_token", "r") as f:
    login(token=f.read().strip())

MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"

# 1. Define the separate configs
t5_quant_config = TransformersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

transformer_quant_config = DiffusersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("Loading Encoders and VAE...")
clip_text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16).to("cuda")
vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae").to("cuda", dtype=torch.bfloat16)

# 2. Apply the Transformers config to T5
print("Loading Quantized T5...")
t5_text_encoder = T5EncoderModel.from_pretrained(
    MODEL_ID, 
    subfolder="text_encoder_2", 
    quantization_config=t5_quant_config,
    torch_dtype=torch.bfloat16
)

# 3. Apply the Diffusers config to the Transformer
print("Loading Quantized Transformer...")
transformer = FluxTransformer2DModel.from_pretrained(
    MODEL_ID,
    subfolder="transformer",
    quantization_config=transformer_quant_config,
    torch_dtype=torch.bfloat16
)

print("Assembling Pipeline...")
pipe = SliderEditFluxKontextPipeline.from_pretrained(
    MODEL_ID,
    vae=vae,
    text_encoder=clip_text_encoder,
    text_encoder_2=t5_text_encoder,
    transformer=transformer,
    torch_dtype=torch.bfloat16
)

checkpoint_dir = "/tmp/eoikonom/SliderEdit/checkpoints/example_training_gstlora_iter500.safetensors"

pipe.load_gstlora(checkpoint_dir)
pipe.loaded_adapter = LoRAAdapterType.GSTLORA

# 4. Test the generation speed
print("Running test generation...")
test_img = Image.new('RGB', (1024, 1024), color = 'white') # Dummy input image
start_time = time.time()

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    output_image = pipe(
        image=test_img,  
        prompt="make the background red",
        generator=torch.Generator().manual_seed(42),
        slider_alpha=0.0,
    ).images[0]

end_time = time.time()
print(f"Generation took: {end_time - start_time:.2f} seconds")
