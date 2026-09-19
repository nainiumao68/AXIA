"""Color palettes and mask colorization for segmentation visualizations."""
import colorsys

import numpy as np

# ISPRS Potsdam / Vaihingen 6-class palette (RGB).
_ISPRS_PALETTE = [
    [255, 255, 255],  # impervious surfaces
    [0, 0, 255],      # building
    [0, 255, 255],    # low vegetation
    [0, 255, 0],      # tree
    [255, 255, 0],    # car
    [255, 0, 0],      # clutter / background
]


def build_class_colors(num_classes, dataset_name=''):
    """Return an RGB palette with ``num_classes`` entries.

    Uses the ISPRS convention for Potsdam/Vaihingen and an evenly spaced HSV
    palette otherwise (index 0 is black).
    """
    if dataset_name.lower() in {'potsdam', 'vaihingen'} and num_classes == 6:
        return [list(c) for c in _ISPRS_PALETTE]
    colors = [[0, 0, 0]]
    for i in range(1, num_classes):
        hue = (i * 360 // num_classes) % 360
        r, g, b = colorsys.hsv_to_rgb(hue / 360.0, 0.8, 0.8)
        colors.append([int(r * 255), int(g * 255), int(b * 255)])
    return colors


def colorize_mask(mask, colors):
    """Map an HxW integer label mask to an HxWx3 color image."""
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(colors):
        out[mask == cls_idx] = color
    return out
