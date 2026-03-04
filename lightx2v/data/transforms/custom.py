import numpy as np
from PIL import Image

def crop_and_resize(image, height, width):
    image = np.array(image)
    image_height, image_width, _ = image.shape
    if image_height / image_width < height / width:
        croped_width = int(image_height / height * width)
        left = (image_width - croped_width) // 2
        image = image[:, left: left+croped_width]
        image = Image.fromarray(image).resize((width, height))
    else:
        croped_height = int(image_width / width * height)
        left = (image_height - croped_height) // 2
        image = image[left: left+croped_height, :]
        image = Image.fromarray(image).resize((width, height))
    return image
