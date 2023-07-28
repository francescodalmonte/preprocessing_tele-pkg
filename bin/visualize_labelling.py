import os
from tkinter import Tk
from tkinter.filedialog import askdirectory

import numpy as np
import matplotlib.patches as patches
from matplotlib import pyplot as plt

from preprocessing_tele.multiChannelImage import multiChannelImage

#####################################################################
#                                                                   #
#   Simple script for visualizing the labelled ROIs directly        #
#   on raw images. Run it from terminal and select the directory    #
#   with the images; corresponding metadata files are supposed      #
#   to be in the same parent-directory.                             #
#                                                                   #
#####################################################################


def draw_ROIs(mcImage, axis):
    """
    """
    # get ROIs
    centers, _ = mcImage.__get_metadata__(scale = 1.)

    # get image
    image = mcImage.__get_diffImage__()

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
