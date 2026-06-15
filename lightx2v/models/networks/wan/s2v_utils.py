import math


def get_size_less_than_area(height, width, target_area, divisor=64):
    if height * width <= target_area:
        max_upper_area = target_area
        min_scale = 0.1
        max_scale = 1.0
    else:
        max_upper_area = target_area
        d = divisor - 1
        b = d * (height + width)
        a = height * width
        c = d**2 - max_upper_area
        min_scale = (-b + math.sqrt(b**2 - 2 * a * c)) / (2 * a)
        max_scale = math.sqrt(max_upper_area / (height * width))

    for i in range(100):
        scale = max_scale - (max_scale - min_scale) * i / 100
        new_height, new_width = int(height * scale), int(width * scale)
        pad_height = (64 - new_height % 64) % 64
        pad_width = (64 - new_width % 64) % 64
        padded_height, padded_width = new_height + pad_height, new_width + pad_width
        if padded_height * padded_width <= max_upper_area:
            return padded_height, padded_width

    aspect_ratio = width / height
    target_width = int((target_area * aspect_ratio) ** 0.5 // divisor * divisor)
    target_height = int((target_area / aspect_ratio) ** 0.5 // divisor * divisor)
    if target_width >= width or target_height >= height:
        target_width = int(width // divisor * divisor)
        target_height = int(height // divisor * divisor)
    return target_height, target_width
