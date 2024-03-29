# input/output utilities

import os
from typing import List

import numpy as np
import cv2 as cv


def read_singleImage(path: str,
                     scale: float = 1.,
                     standardize: bool = False) -> np.array:
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
                  suffix: str = "bmp") -> np.array:
    """Read multiple images from directory.

        Parameters
    ----------
    path: absolute path to images directory.
    scale: rescaling factor (default = 1).
    suffix: allowed image format (default = "bmp")

    Returns
    ----------
    images: array containing the images.

    """
    path = os.path.abspath(path)
    assert os.path.isdir(path), f"No directory found at {path}"

    imgs = []
    for file in sorted(os.listdir(path)):
        if file.endswith(suffix):
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
    data: coordinates ([X, Y, W, H, ID], recaled).
    imshape: images shape ([Width, Height], rescaled).
    """

    path = os.path.abspath(path)
    
    if not os.path.isfile(path): 
        return None
    
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

                if ("ImageHeight" in line[0]): imH = int(float(line[1])*scale)
                if ("ImageWidth" in line[0]): imW = int(float(line[1])*scale)


    data = np.column_stack((X, Y, W, H, i)).astype(int)

    return data, [imH, imW]
        


def listRawDir(path: str,
               exclusion: List[str,] = []) -> List[str]:
    """Returns a list of the raw images names in a directory.

    Parameters
    ----------
    path: absolute path to root directory.
    exclusion: list of names to be excluded.

    Returns
    ----------
    names: list of raw images names.
    """

    names = []
    for f in os.listdir(os.path.abspath(path)):
        if f.endswith(".obj"):
            n = f.rsplit(".", 1)[-2]
            if n not in exclusion:
                names.append(n)

    return names


def saveInfo(config: dict,
              save_path: str):
    """Save dataset info to .txt file.
    """
    file_path = os.path.join(save_path, "info.txt")
    with open(file_path,"w") as file:
        for k in config.keys():
            file.write(k + " >>> " + str(config[k]) + "\n")