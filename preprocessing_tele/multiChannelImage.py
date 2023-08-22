# multiChannelImage class

import os 
import numpy as np
import cv2 as cv

from .IO import read_dirImage, read_metadataFile, read_singleImage
from .image_processing import randomFlip, randomShift, cropImage

class multiChannelImage():
    """Base class for multichannel images"""

    def __init__(self, name: str, rootpath: str):

        self.name = name
        self.rootpath = rootpath
        self.imdirPath = os.path.join(rootpath, name + ".obj")
        self.metadataPath = os.path.join(rootpath, name + ".dat")
        self.maskPath = os.path.join(rootpath, name + "_M.png")


    def __get_images__(self, scale: float = 1, suffix: str = "bmp"):
        """
        Read raw images from file.
        """
        return read_dirImage(self.imdirPath,
                             scale=scale,
                             suffix = "bmp")

    def __get_diffImage__(self, scale: float = 1,
                          suffix: str = "bmp",
                          minuend = 3,
                          subtrahend = 2):
        """
        Read raw images and return grazing lights difference image.
        """
        imgs = self.__get_images__(scale = scale)
        diffImage = imgs[minuend].astype(float) - imgs[subtrahend].astype(float)
        diffImage = (diffImage + 128.).astype(int)
        return diffImage

    def __get_metadata__(self, scale: float = 1):
        return read_metadataFile(self.metadataPath,
                                 scale=scale)

    def __get_anomalousMask__(self, scale: float = 1):
        """
        Read binary anomaly mask from file.
        """

        return read_singleImage(self.maskPath, scale = scale)

    def __get_goodMask__(self, scale: float = 1., size: int = 224):
        """
        Create a mask of the nominal area.
        (this function is used to generate random centers for good crops).

        """    
        centers, imshape = self.__get_metadata__(scale = scale)
        mask = np.ones(imshape)

        # margin
        l = int(size//2) + 5

        mask[:l, :] = 0
        mask[-l:, :] = 0
        mask[:, :l] = 0
        mask[:, -l:] = 0

        if centers is not None:
            for c in centers:
                x, y, w, h, _ = c.astype(int)

                top = np.max([0, y-h-l])
                bottom = np.min([imshape[0], y+h+l])
                left = np.max([0, x-w-l])
                right = np.min([imshape[1], x+w+l])

                mask[top : bottom, left : right] = 0

        return mask


    
    def __get_randomCenters__(self, mask, N):
        """
        Generate random centers coordinates given a binary 2D mask.
        (This function is used to generate the set of good crops).

        """        
        all_coords = np.argwhere(mask) # list of (y, x) values

        idxs =  np.random.randint(0, len(all_coords), size=N)
        
        centers = all_coords[idxs][:, ::-1] # list of (x, y) values
        
        extra_col = -1*np.ones(len(centers))
        padded = np.c_[ centers, extra_col, extra_col, extra_col ]

        return padded.astype(int)



    def fetch_goodCrops(self,
                        N,
                        scale = 1.,
                        size = 224,
                        rand_flip = False,
                        normalize = True,
                        gauss_blur = None,
                        mode = "diff"):
        """
        Create a set of good crops using randomly generated coordinates.
        (This method is based on the cropImage() function).

        Parameters
        ----------
        N: number of crops to be fetched.

        Returns
        ----------

        """

        # images
        if mode == "diff":
            image = self.__get_diffImage__(scale = scale)
        elif mode in [0,1,2,3,4]:
            image = self.__get_images__(scale = scale)[mode]
        else:
            raise(ValueError("Invalid argument: mode"))
        
        # crops coordinates
        mask = self.__get_goodMask__(scale = scale, size = size)
        centers = self.__get_randomCenters__(mask, N = N)

        # image for binary masks
        mask = self.__get_anomalousMask__(scale = scale)
        image = np.stack((image, mask), axis = 2)

        # run cropImage()
        crops, centers = cropImage(image, centers, size = size,
                                   rand_flip = rand_flip,
                                   rand_shift = False,
                                   normalize = normalize,
                                   gauss_blur = gauss_blur)

        return crops, centers



    def fetch_anomalousCrops(self, scale = 1., size = 224,
                             rand_shift = False,
                             rand_flip = False,
                             gauss_blur = None,
                             normalize = True,
                             mode = "diff"
                             ):
        """
        Create a set of anomalous crops using the coordinates from the metadata file.
        (This method is based on the cropImage() function).

        Parameters
        ----------


        Returns
        ----------

        """

        # images
        if mode == "diff":
            image = self.__get_diffImage__(scale = scale)
        elif mode in [0,1,2,3,4]:
            image = self.__get_images__(scale = scale)[mode]
        else:
            raise(ValueError("Invalid argument: mode"))

        # crops coordinates
        centers, _ = self.__get_metadata__(scale = scale)
        centers = centers[centers[:,4]<1] 

        # image for binary masks
        mask = self.__get_anomalousMask__(scale = scale)
        image = np.stack((image, mask), axis = 2)

        # run cropImage()
        crops, centers = cropImage(image, centers, size = size,
                                   rand_flip = rand_flip,
                                   rand_shift = rand_shift,
                                   normalize = normalize,
                                   gauss_blur = gauss_blur)

        return crops, centers