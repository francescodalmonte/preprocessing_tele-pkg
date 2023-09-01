# functions for generic image manipulation

import numpy as np
import os
import cv2 as cv
from scipy.ndimage import gaussian_filter


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
              rand_flip: bool = False,
              normalize: bool = True,
              gauss_blur: float = None,
              min_area: int = -1) -> np.array:
    """
    Returns a set of fixed-size square crops of an input image.

    Parameters
    ----------
    image: image to be cropped
           (may have 1 or 2 channels. In the latter case, the second channel is 
           supposed to represent the anomaly binary mask, and its crops are returned
           as well).
    centers: set of centers coordinates of the crops.
    size: side length of the crop (pixels. Default = 224)
    rand_shift: perform random shift of the centers (default = False).
    rand_flip: perform random flip of the crops (default = False).
    normalize: whether to normalize or not the final crop (default = True).
    gauss_blur: size of gaussian blurring kernel (default = None, i.e. no blurring).
    min_area: minimum defect area to include the crop (default = -1, i.e. all crops are included)

    Returns
    ----------
    crops_set: set of crops.

    """
    crops_set = []
    centers_set = []

    for c in centers:
        x, y = c[:2].astype(int)
        l = int(size/2)

        if rand_shift : x, y = randomShift(x, y, l-25) 
        crop = np.array(image[y-l : y+l, x-l : x+l])
        if rand_flip : crop = randomFlip(crop) 

        if len(crop.shape) == 3: # images WITH binary mask case
            if crop.shape[:2] == (size, size):
                
                if gauss_blur is not None:
                    # gaussian blurring
                    crop[:,:,0] = gaussian_filter(crop[:,:,0], sigma = gauss_blur)

                if normalize:
                    # normalization at single crop level
                    crop[:,:,0] = (crop[:,:,0] - np.mean(crop[:,:,0])) + 128

                mask_area = np.sum(crop[:,:,1]>0)
                if mask_area > min_area:
                    crops_set.append(crop)
                    centers_set.append([x, y])

        if len(crop.shape) == 2: # images WITHOUT binary mask case
            if crop.shape == (size, size):
                
                if gauss_blur is not None:
                    # gaussian blurring
                    crop = gaussian_filter(crop, sigma = gauss_blur)

                if normalize:
                    # normalization at single crop level
                    crop = (crop - np.mean(crop)) + 128

                crops_set.append(crop)
                centers_set.append([x, y])

    return np.array(crops_set), np.array(centers_set)


def tileImage(image: np.array,
              size: int = 224,
              overlap: int = 0,
              normalize: bool = True,
              gauss_blur: float = None) -> np.array:
    """
    Create a regular tiling of an image (uses: cropImage function).
    """
    imshape = np.shape(image)
    y = np.arange(size//2 + 1, imshape[0] - size//2 - 1, size - overlap)
    x = np.arange(size//2 + 1, imshape[1] - size//2 - 1, size - overlap)
    grid = np.meshgrid(x, y)
    coords = np.vstack([grid[0].ravel(), grid[1].ravel()]).T
    
    tiles_set, centers_set = cropImage(image = image,
                                       centers = coords,
                                       size = size,
                                       rand_shift = False,
                                       rand_flip = False,
                                       normalize = normalize,
                                       gauss_blur = gauss_blur)
    
    return np.array(tiles_set[:,:,:]), np.array(centers_set)


def saveCrops(save_to, crops_set, centers_set, prefix = "", suffix = ""):
    """
    """

    assert os.path.isdir(save_to), f"No dir found at {save_to}"
    assert len(crops_set)==len(centers_set), "len() of crops set not matching centers set"

    for crop, coords in zip(crops_set, centers_set):
        filename = prefix + f"C({coords[0]}-{coords[1]})" + suffix + ".png" 
        path = os.path.join(save_to, filename)
        
        img = np.array([crop, crop, crop]).transpose(1,2,0)
        cv.imwrite(path, img)
