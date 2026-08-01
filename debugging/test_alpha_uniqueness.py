import torch
from PIL import Image
import numpy as np

# Simulate what your generator and pipeline receive during an alpha loop
def mock_pipeline_behavior(image, prompt, slider_alpha):
    # Simulate an edit effect by modifying pixel tint or intensity based on alpha
    img_np = np.array(image).astype(np.float32)
    
    # Scale changes based on slider_alpha (e.g., alpha 1.0 brightens, -1.0 darkens)
    factor = 1.0 + (float(slider_alpha) * 0.2)
    modified_np = np.clip(img_np * factor, 0, 255).astype(np.uint8)
    
    return Image.fromarray(modified_np)

def test_sweep():
    # Create a dummy test image
    base_img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    alpha_values = [1.0, 0.5, 0.0, -0.5, -1.0]
    
    generated_variants = []
    print("Testing alpha sweep uniqueness...")
    
    for alpha in alpha_values:
        out_img = mock_pipeline_behavior(base_img, "make it brighter", alpha)
        generated_variants.append(np.array(out_img))
        print(f"Alpha {alpha:4.1f} -> Pixel mean value: {np.mean(out_img):.2f}")

    # Verify that no two consecutive outputs are identical
    for i in range(len(generated_variants) - 1):
        assert not np.array_equal(generated_variants[i], generated_variants[i+1]), \
            f"❌ Failure: Alpha steps {alpha_values[i]} and {alpha_values[i+1]} produced identical images!"

    print("\n✅ Success: All alpha steps produce uniquely modified outputs!")

if __name__ == "__main__":
    test_sweep()