# functions for generic image manipulation

import numpy as np

def randomFlip(crop):
    """
    """
    if np.random.rand() > .5 : crop = crop[:, ::-1]
    if np.random.rand() > .5 : crop = crop[::-1, :]
    return crop
    


def randomShift(x, y, max_shift = 100):
    """
    """
    x += np.random.randint(-max_shift, max_shift)
    y += np.random.randint(-max_shift, max_shift)
    return x, y


def cropImage(image, centers, size = 224,
              rand_shift = False, rand_flip = False):
    """
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