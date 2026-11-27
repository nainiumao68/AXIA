import numpy as np


def build_class_colors(num_classes, dataset_name=''):
    import colorsys
    if dataset_name.lower() in {'potsdam', 'vaihingen'} and num_classes == 6:
        return [[255, 255, 255], [0, 0, 255], [0, 255, 255],
                [0, 255, 0], [255, 255, 0], [255, 0, 0]]
    colors = [[0, 0, 0]]
    for i in range(1, num_classes):
        colors.append([int(v * 255) for v in colorsys.hsv_to_rgb(i / num_classes, 0.8, 0.8)])
    return colors


def colorize_mask(mask, colors):
    result = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for index, color in enumerate(colors):
        result[mask == index] = color
    return result
