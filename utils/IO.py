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

    def __get_goodMask__(self, scale: float = 1., size: int = 224):
        """
        Create a mask of the anomalous area.
        (this function is used to generate random centers for good crops).

        """    
            #######################################
        ############### da implementare ################
            #######################################

        return 0
    
    def __get_randomCenters__(self, mask):
        """
        Generate random centers coordinates given a binary 2D mask.
        (This function is used to generate the set of good crops).

        """
                ############################################
        ############### da implementare #####################
            #######################################

        return 0


    def cropImage(self, image, centers, size = 224):
        """

        """
        crops = []
        for c in centers:
            x, y, _, _, _ = c.astype(int)
            l = int(size/2)
            crops.append(image[y-l : y+l,
                               x-l : x+l])

        return crops



    def fetch_goodCrops(self, N, scale = 1., size = 224):
        """
        Create a set of good crops using randomly generated coordinates.
        (This method is based on the cropImage() function).

        Parameters
        ----------
        N: number of crops to be fetched.

        Returns
        ----------
        """

        # __get_goodMask()
        # __get_randomCenters()
        # cropImage
        
          ##########################################
        ########## da implementare ###################
          ########################################

        return 0 



    def fetch_anomalousCrops(self, scale = 1., size = 224):
        """
        Create a set of anomalous crops using the coordinates from the metadata file.
        (This method is based on the cropImage() function).

        Parameters
        ----------


        Returns
        ----------

        """
        imgs = self.__get_images__(scale = scale)
        centers = self.__get_metadata__(scale = scale)
        
        image = imgs[0].astype(float) - imgs[1].astype(float)
        image = (image + 128.).astype(int)

        crops = self.cropImage(image, centers, size = size)

        return crops
        

