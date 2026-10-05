import numpy as np
from PIL import Image


def detect_ingrs(recipe, vocab):
    try:
        ingr_names = [ingr['text'] for ingr in recipe['ingredients'] if ingr['text']]
    except (KeyError, TypeError):
        ingr_names = []
        print("Could not load ingredients! Moving on...")

    detected = set()
    for name in ingr_names:
        name = name.replace(' ', '_')
        name_ind = vocab.get(name)
        if name_ind:
            detected.add(name_ind)

    return list(detected) + [vocab['</i>']]


def _to_rgb_array(img):
    """PIL image -> uint8 HxWx3 array (grayscale and RGBA are converted)."""
    return np.asarray(img.convert('RGB'))


def process_image(impath, imsize):
    """Load an image and scale it so that its shorter side is `imsize`."""
    try:
        with Image.open(impath) as im:
            W0, H0 = im.size
            scale = float(imsize) / min(H0, W0)
            img = _to_rgb_array(im.convert('RGB').resize((round(W0 * scale), round(H0 * scale)), Image.BILINEAR))
        fail = 0
    except Exception:
        print("Could not load image...Using black one instead.")
        img = np.zeros((imsize, imsize, 3))
        fail = 1

    return img, fail


def read_image(filename):
    with Image.open(filename) as im:
        return _to_rgb_array(im.convert('RGB').resize((224, 224), Image.BILINEAR))
