import math


def resize_to_max_side(image, max_side):
    width, height = image.size
    if width >= height:
        new_width = max_side
        new_height = int(max_side * height / width)
    else:
        new_height = max_side
        new_width = int(max_side * width / height)
    return image.resize((new_width, new_height))


def resize_to_target_area(image, target_area):
    w, h = image.size
    ratio = w / h
    new_w = round(math.sqrt(target_area * ratio) / 16) * 16
    new_h = round(math.sqrt(target_area / ratio) / 16) * 16
    # Scale so that both dimensions are at least the target size, then crop.
    scale = max(new_w / w, new_h / h)
    scaled_w = round(w * scale)
    scaled_h = round(h * scale)
    image = image.resize((scaled_w, scaled_h), resample=3)  # BICUBIC=3
    # Center crop to exact (new_w, new_h)
    left = (scaled_w - new_w) // 2
    top = (scaled_h - new_h) // 2
    image = image.crop((left, top, left + new_w, top + new_h))
    return image
