import numpy as np
from preprocessing import list_dir_files, load
from PIL import Image
from skimage.segmentation import flood
from skimage.feature import canny
from skimage.measure import shannon_entropy
from skimage.transform import radon
import cv2 as cv

def calculate(dirpath: str):
    for filepath in list_dir_files(dirpath):
        calculate_single(filepath)

def calculate_single(filepath: str):
    img = Image.open(filepath).convert("RGB")
    img_linear, img_srgb = load(img)
    gray = np.mean(img_linear, axis=2)

    jpeg_size = calculate_jpeg_size(img)

    object_size_masked, object_size_flooded = calculate_object_size(img_srgb)
    if abs(object_size_masked - object_size_flooded) > 200:
        print(f"Significant difference in flood fill vs np masking technique for {filepath}")

    contrast = np.std(gray)
    luminance = np.mean(gray)

    colorfulness = calculate_colorfulness(img_srgb)

    hue, sat, val = calculate_hsv(img_srgb)

    entropy = shannon_entropy(gray)

    edges = calculate_edges(gray)

    symmetry = calculate_symmetry(gray)

def calculate_jpeg_size(img: Image.Image) -> int:
    width, height = img.size
    return width*height

def calculate_object_size(img: np.ndarray) -> tuple[int, int]:
    # Get edge colours and choose if possible with high confidence
    h, w = img.shape[:2]

    background_color = None

    mask_value = -1
    flood_value = -1

    top_left = img[0, 0]
    top_right = img[0, w - 1]
    bottom_left = img[h - 1, 0]
    bottom_right = img[h - 1, w - 1]

    corners = (top_left, top_right, bottom_left, bottom_right)
    if not all(np.sqrt(np.sum((a-b)**2)) < 5 for a in corners for b in corners):
        top_center = img[0, w // 2]
        bottom_center = img[h - 1, w // 2]
        left_center = img[h // 2, 0]
        right_center = img[h // 2, w - 1]

        test_points = corners + (top_center, bottom_center, left_center, right_center)
        pairs = [(a,b) for a in test_points for b in test_points if a is not b]
        similar = sum(np.sqrt(np.sum((a-b)**2)) < 5 for a, b in pairs)
        ratio = similar / len(pairs)
        if ratio > 0.7:
            background_color = next(
                a for i, a in enumerate(test_points)
                if sum(np.sqrt(np.sum((a-b)**2)) < 5
                    for j, b in enumerate(test_points) if i != j)
                / (len(test_points) - 1) >= 0.7
            )
    else:
        background_color = top_left

    if background_color is not None:

        numpy_mask = np.all(img == background_color, axis=2)
        mask_value = w*h - np.sum(numpy_mask)

        flood_mask = np.zeros((h, w), bool)
        for seed in [(0,0), (0,w-1), (h-1,0), (h-1,w-1)]:
            flood_mask |= flood(img, seed, tolerance=5)
        flood_value = w*h - np.sum(flood_mask)

    return mask_value, flood_value

def calculate_colorfulness(img: np.ndarray) -> float:
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    rg = R - G
    yb = 0.5*(R + G) - B
    mu = np.sqrt(np.mean(rg)**2+np.mean(yb)**2)
    sigma = np.sqrt(np.std(rg)**2+np.std(yb)**2)
    return sigma + 0.3 * mu

def calculate_hsv(img: np.ndarray) -> tuple[int, int, int]:
    hsv = cv.cvtColor((img * 255).astype(np.uint8), cv.COLOR_RGB2HSV)
    return hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

def calculate_edges(gray: np.ndarray) -> float:
    edges = canny(gray)
    return np.sum(edges) / edges.size

def calculate_symmetry(gray: np.ndarray) -> float:
    angles = np.linspace(0, 180, 180, endpoint=False)
    sinogram = radon(gray, theta=angles)
    return np.mean(sinogram)

def calculate_MSER(gray: np.ndarray) -> float:
    gray_uint8 = (gray * 255).astype(np.uint8)
    mser = cv.MSER_create()
    regions, _ = mser.detectRegions(gray_uint8)
    return len(regions)

def calculate_SIFT(gray: np.ndarray) -> float:
    gray_uint8 = (gray * 255).astype(np.uint8)
    sift = cv.SIFT_create()
    keypoints, _ = sift.detectAndCompute(gray_uint8, None)
    return len(keypoints)


