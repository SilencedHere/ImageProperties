
import numpy as np
from PIL import Image

def srgb_to_linear(img: Image.Image) -> np.ndarray:
    rgb = np.array(img).astype(np.float32) / 255.0
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4
    )
    return linear

def load(filepath: str) -> (np.ndarray, np.ndarray):
    img = Image.open(filepath).convert("RGB")
    srgb = np.array(img) / 255.0
    linear = srgb_to_linear(img)