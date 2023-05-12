import os
from tkinter import Tk
from tkinter.filedialog import askdirectory

import numpy as np
import matplotlib.patches as patches
from matplotlib import pyplot as plt

from utils.multiChannelImage import multiChannelImage

def draw_ROIs(mcImage, axis):
    """
    """
    # get ROIs
    centers, _ = mcImage.__get_metadata__(scale = 1.)

    # get image
    imgs = mcImage.__get_images__(scale = 1.)   
    image = imgs[3].astype(float) - imgs[2].astype(float)
    image = (image + 128.).astype(int).T

    # draw
    axis.imshow(image, cmap="Greys")
    for x, y, w, h, id in centers:
        rect = patches.Rectangle((y-h//2, x-w//2), h, w,
                                 linewidth = 2,
                                 edgecolor = 'r',
                                 facecolor = 'none')
        axis.add_patch(rect)

    return axis


if __name__ == "__main__":

    print("Select the folder (.obj) containing the images... (the .dat corresponding file is expected be in the same root directory)")
    
    Tk().withdraw()
    filename = askdirectory()

    root = os.path.dirname(filename)
    name = os.path.splitext(os.path.basename(filename))[0]

    print(root)
    print(name)

    mcImage = multiChannelImage(name, root)

    fig, ax = plt.subplots(figsize = (10,4))
    draw_ROIs(mcImage, ax)

    plt.show()
