# functions for generic image manipulation

def randomFlip(image):
    """
    """
    


def randomShift():
    """
    """


def cropImage(image, centers, size = 224,
              rand_shift = False, rand_flip = False):
    """
    """
    crops = []

    for c in centers:
        x, y, h, w, id = c.astype(int)

        l = int(size/2)

        im = image[y-l : y+l, x-l : x+l]
            

        crops.append(im)

    return crops