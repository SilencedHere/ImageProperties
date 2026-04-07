from dataclasses import dataclass

@dataclass
class ImageMetrics:
    filename: str
    jpeg_size: int
    object_size_masked: int
    object_size_flooded: int
    contrast: float
    bg_contrast: float
    luminance: float
    luminance_linear: float
    colorfulness: float
    hue: int
    sat: float
    val: float
    entropy: float
    edges: float
    symmetry: float
    MSER: float
    SIFT: float
    spectral_energy: float
    low_band: float
    mid_band: float
    high_band: float
    high_spatial_frequencies: float