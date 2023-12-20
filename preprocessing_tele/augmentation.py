import os
import numpy as np
import cv2 as cv


def load_blending_mask(path: str,
                       alpha: float = 1.,
                       size: int = 224):
    mask = cv.GaussianBlur(cv.imread(path), (9,9), 0)
    mask = cv.resize(mask, dsize = size)
    mask = mask.astype(float)/255.
    return mask*alpha


def load_defect(path: str,
                size: int = 224):
    defect = cv.imread(path)
    defect = cv.resize(defect, dsize = size)
    return defect



def blend(defect_path: np.ndarray,
          mask_path: np.ndarray,
          image_path: np.ndarray,
          alpha: float = 1.):
    image = cv.imread(image_path)
    size = image.shape[:2]
    mask = load_blending_mask(mask_path, alpha = alpha, size = size)
    defect = load_defect(defect_path, size = size)
        

    reciprocal_mask = (np.ones_like(mask) - mask)
    mixed = mask*defect + reciprocal_mask*image

    return np.clip(mixed, 0, 255).astype(int)

