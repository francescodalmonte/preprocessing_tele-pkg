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
    


def randomShift(x, y, max_shift,
                limitsx = [0, 2048],
                limitsy = [0, 7750]):
    """
    Performs a random shift of an input point [x,y] (max displacement = max_shift).
    """

    x += np.random.randint(np.max([-max_shift, (limitsx[0]-x)]), np.min([max_shift, (limitsx[1]-x)]))
    y += np.random.randint(np.max([-max_shift, (limitsy[0]-y)]), np.min([max_shift, (limitsy[1]-y)]))

    return x, y



def cropImage(image: np.array,
              centers: np.array,
              size: int = 224,
              rand_shift: bool = False,
              rand_flip: bool = False,
              normalize: bool = True,
              gauss_blur: float = None,
              min_area: int = -1,
              mask_threshold: list[int,] = [0, 256],
              saturate_mask = True) -> np.array:
    """
    Returns a set of fixed-size square crops of an input image.

    Parameters
    ----------
    image: image to be cropped
           (may have 3 or 4 channels. In the latter case, the fourth channel is 
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

    assert size%2 == 0, "size must be an even number"

    for c in centers:
        x, y = c[:2].astype(int)
        l = int(size/2)

        if rand_shift :
            x, y = randomShift(x, y, l-25,
                               limitsx = [size//2, image.shape[1]-size//2],
                               limitsy = [size//2, image.shape[0]-size//2]) 

        crop = np.array(image[y-l : y+l, x-l : x+l])
        if rand_flip : crop = randomFlip(crop) 

        if gauss_blur is not None:
            if gauss_blur>0.:
                # gaussian blurring
                crop[:,:,:3] = gaussian_filter(crop[:,:,:3], sigma = gauss_blur, axes=(0,1))

        if normalize:
            # normalization at single crop level
            crop[:,:,:3] = (crop[:,:,:3] - np.mean(crop[:,:,:3])) + 128

        if crop.shape == (size, size, 4): # crops WITH mask
            # threshold mask to only consider certain defects
            crop[:,:,3][crop[:,:,3]<=mask_threshold[0]] = 0
            crop[:,:,3][crop[:,:,3]>=mask_threshold[1]] = 0
            if saturate_mask:
                crop[:,:,3][crop[:,:,3]>0] = 255
            mask_area = np.sum(crop[:,:,3]>0)
            if mask_area > min_area:
                crops_set.append(crop)
                centers_set.append([x, y])
        elif crop.shape == (size, size, 3): # crops WITHOUT mask
            crops_set.append(crop)
            centers_set.append([x, y])

    return np.array(crops_set), np.array(centers_set)


def tileImage(image: np.array,
              size: int = 224,
              overlap: int = 0,
              normalize: bool = True,
              gauss_blur: float = None,
              min_area: int = -1,
              mask_threshold: list[int,] = [0, 256],
              saturate_mask = True) -> np.array:
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
                                       gauss_blur = gauss_blur,
                                       min_area = min_area,
                                       mask_threshold = mask_threshold,
                                       saturate_mask = saturate_mask
                                       )
    
    return np.array(tiles_set[:,:,:]), np.array(centers_set)



def CLAHEtransformation(input: np.ndarray,
                        **kwargs):
    """
    Apply Contrast Limited Adaptive Histogram Equalization tranform to 
    an input single-channel image.
    (https://en.wikipedia.org/wiki/Adaptive_histogram_equalization)

    """
    clahe = cv.createCLAHE(**kwargs)
    return clahe.apply(input.astype(np.uint8))



def CEtransformation(input: np.ndarray,
                    kernel_size: list = [1, 7]):
    """
    Apply Contrast Enhancement transformation to an input single-channel image.
    """
    blur1 = cv.medianBlur(input.astype(np.uint8), kernel_size[0])
    blur2 = cv.medianBlur(input.astype(np.uint8), kernel_size[1])

    contrast = blur1.astype(float) - blur2.astype(float)
    output = np.clip(input+contrast*2, 0, 255).astype(np.uint8)

    return output




def saveCrops(save_to, crops_set, centers_set, prefix = "", suffix = "", mode = None):
    """
    NB: for modes "pseudo_color" or "color_map" the crops are supposed to be grayscale
    (i.e. 3 identical channels)
    """

    assert os.path.isdir(save_to), f"No dir found at {save_to}"
    assert len(crops_set)==len(centers_set), "len() of crops set not matching centers set"

    for crop, coords in zip(crops_set, centers_set):
        filename = prefix + f"C({coords[0]}-{coords[1]})" + suffix + ".png" 
        path = os.path.join(save_to, filename)
        
        if mode == "pseudo_color":
            ch1 = crop
            ch2 = CLAHEtransformation(ch1, clipLimit=3.0, tileGridSize=(5,5))
            ch3 = CLAHEtransformation(ch1, clipLimit=3.0, tileGridSize=(8,8))
            img = np.array([ch1, ch2, ch3]).transpose(1,2,0)
        elif mode == "color_map":
            img = cv.applyColorMap(crop, cv.COLORMAP_JET)
        else:
            img = crop
        
        cv.imwrite(path, img)




def localContrastCorrection(image, sigma = 10):
        """
        Performs a local contrast normalization of an input image by dividing 
        it by its gaussian blurred version.
        (requires float input image)
        """

        # blur
        image_resized = cv.resize(image.copy(), (0,0), fx=0.25, fy=0.25, interpolation = cv.INTER_AREA)
        image_filtered = gaussian_filter(image_resized,
                                         sigma = (sigma, sigma),
                                         axes=(0,1),
                                         truncate = 3)
        image_filtered = cv.resize(image_filtered, (image.shape[1], image.shape[0]), interpolation = cv.INTER_LINEAR)
        image_filtered = np.clip(image_filtered, 1., 255.) # avoid division by zero

        # normalize
        image_n = (image/image_filtered)

        # back to 0-255 range
        image_n = (image_n-1)*128. + 128.

        return image_n

