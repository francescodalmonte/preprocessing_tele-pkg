# functions for generic image manipulation

import numpy as np

def randomFlip(crop):
    """
    Performs random horizontal/vertical flip of an input image (with p = 0.5).
    """
    if np.random.rand() > .5 : crop = crop[:, ::-1]
    if np.random.rand() > .5 : crop = crop[::-1, :]

    return crop
    


def randomShift(x, y, max_shift = 100):
    """
    Performs a random shift of an input point [x,y] (max displacement = max_shift).
    """
    x += np.random.randint(-max_shift, max_shift)
    y += np.random.randint(-max_shift, max_shift)
    
    return x, y


def cropImage(image: np.array,
              centers: np.array,
              size: int = 224,
              rand_shift: bool = False,
              rand_flip: bool = False) -> np.array:
    """
    Returns a set of fixed-size square crops of an input image.

    Parameters
    ----------
    image: image to be cropped.
    centers: set of centers coordinates of the crops.
    size: side length of the crop (pixels. Default = 224)
    rand_shift: perform random shift of the centers (default = False).
    rand_flip: perform random flip of the crops (default = False).

    Returns
    ----------
    crops_set: set of crops.

    """
    crops_set = []

    for c in centers:
        x, y, _, _, _ = c.astype(int)
        l = int(size/2)

        if rand_shift : x, y = randomShift(x, y, l-25) 
        crop = image[y-l : y+l, x-l : x+l]
        if rand_flip : crop = randomFlip(crop) 

        crops_set.append(crop)

    return crops_set