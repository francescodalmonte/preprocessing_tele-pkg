# input/output utilities

import os
import re
from typing import Dict

import numpy as np
import cv2 as cv


def read_singleImage(path: str,
                     scale: float = 1.) -> np.array:
    """Read an image from file.

    Parameters
    ----------
    path: absolute path to image file.
    scale: rescaling factor (default = 1).

    Returns
    ----------
    images: array containing the image.

    """

    path = os.path.abspath(path)
    assert os.path.isfile(path), f"No file found at {path}"

    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, dsize = None, dst = None, fx=scale, fy=scale)

    return np.array(img)




def read_dirImage(path: str,
                  scale: float = 1.,
                  format: str = "bmp") -> np.array:
    """Read multiple images from directory.

        Parameters
    ----------
    path: absolute path to images directory.
    scale: rescaling factor (default = 1).
    format: allowed image format (default = "bmp")

    Returns
    ----------
    images: array containing the images.

    """
    path = os.path.abspath(path)
    assert os.path.isdir(path), f"No directory found at {path}"

    imgs = []
    for file in sorted(os.listdir(path)):
        if file.endswith(format):
            imgs.append(read_singleImage(os.path.join(path, file),
                                         scale = scale))

    return np.asarray(imgs)




def read_metadataFile(path: str,
                      scale: float = 1.) -> np.array:
    """Read metadata from file.

    Parameters
    ----------
    path: absolute path to file.
    scale: rescaling factor (default = 1).

    Returns
    ----------
    data: metadata array ([X, Y, W, H, ID]).
    """

    path = os.path.abspath(path)
    assert os.path.isfile(path), f"No file found at {path}"

    X, Y, W, H, i = [], [], [], [], []

    with open(path) as file:
        for line in file:
            line = line.split("=")
            if len(line)==2:
                if(".CenterX" in line[0]): X.append(float(line[1])*scale)
                if(".CenterY" in line[0]): Y.append(float(line[1])*scale)
                if(".Width" in line[0]): W.append(float(line[1])*scale)
                if(".Height" in line[0]): H.append(float(line[1])*scale)
                if(".ClassId" in line[0]): i.append(float(line[1]))

    data = np.column_stack((X, Y, W, H, i))

    return data


class multiChannelImage():
    """Base class for multichannel images"""

    def __init__(self, name: str, rootpath: str):

        self.name = name
        self.metadataPath = os.path.join(rootpath, name + ".dat")
        self.imdirPath = os.path.join(rootpath, name + ".obj")


    def __get_images__(self, scale: float = 1, format: str = "bmp"):
        return read_dirImage(self.imdirPath,
                             scale=scale,
                             format = "bmp")

    def __get_metadata__(self, scale: float = 1):
        return read_metadataFile(self.metadataPath,
                                 scale=scale)


    def __get_crops__(self, scale = 1., preprocess = "DIFF"):
        """
        """
        imgs = self.__get_images__(scale = scale)
        metadata = self.__get_metadata__(scale = scale)

        crops = []

        if preprocess=="DIFF":
            image = imgs[0].astype(float) - imgs[1].astype(float)
            image = (image + 128.).astype(int)


            for c in metadata:
                x, y, h, w, _ = c.astype(int)
                crops.append(image[y-h*2:y+h*2, x-w*2:x+w*2])

        return crops