def resize_to_max_side(image, max_side):
    width, height = image.size
    if width >= height:
        new_width = max_side
        new_height = int(max_side * height / width)
    else:
        new_height = max_side
        new_width = int(max_side * width / height)
    return image.resize((new_width, new_height))


def center_crop_to_ratio(image, ratio):
    width, height = image.size
    target = ratio[0] / ratio[1]
    current = width / height
    if current > target:
        new_width = int(height * target)
        left = (width - new_width) // 2
        return image.crop((left, 0, left + new_width, height))
    new_height = int(width / target)
    top = (height - new_height) // 2
    return image.crop((0, top, width, top + new_height))
