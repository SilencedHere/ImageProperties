import numpy as np
from preprocessing import list_dir_files, load
from PIL import Image


def calculate(dirpath: str):
    for filepath in list_dir_files(dirpath):
        calculate_single(filepath)

def calculate_single(filepath: str):
    img = Image.open(filepath).convert("RGB")
    img_linear, img_srgb = load(img)

def jpeg_size(img: Image.Image):
    width, height = img.size
    return width*height

