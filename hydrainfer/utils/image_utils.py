import numpy as np
from PIL import Image

def make_random_image(height: int, width: int, n_channel: int) -> Image:
    random_array = np.random.randint(0, 256, (height, width, n_channel), dtype=np.uint8)
    image = Image.fromarray(random_array)
    return image