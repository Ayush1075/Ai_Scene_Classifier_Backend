"""
Runs during `render build` — downloads and caches the model weights
so the first /analyze request doesn't time out downloading them.
"""
import os
os.environ.setdefault("TORCH_HOME", os.path.expanduser("~/.cache/torch"))

print("Pre-downloading DeepLabV3-MobileNetV3 weights...")
print(f"Using torch cache at: {os.environ['TORCH_HOME']}")

# Force torchvision to download + cache the model
from torchvision import models
model = models.segmentation.deeplabv3_mobilenet_v3_large(weights="DEFAULT")
del model

print("Model weights cached successfully.")
