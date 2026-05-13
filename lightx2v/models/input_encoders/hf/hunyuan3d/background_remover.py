from PIL import Image
from rembg import remove, new_session


class BackgroundRemover:
    def __init__(self):
        self.session = new_session()

    def __call__(self, image: Image.Image) -> Image.Image:
        return remove(image, session=self.session, bgcolor=[255, 255, 255, 0])
