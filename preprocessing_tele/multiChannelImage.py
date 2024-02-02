# multiChannelImage class

import os 
import numpy as np
import cv2 as cv
import json

from .IO import read_dirImage, read_metadataFile, read_singleImage
from .image_processing import randomFlip, randomShift, cropImage


from matplotlib import pyplot as plt

class multiChannelImage():
    """Base class for multichannel images"""

    def __init__(self, name: str, rootpath: str):

        self.name = name
        self.rootpath = rootpath
        self.imdirPath = os.path.join(rootpath, name + ".obj")
        self.metadataPath = os.path.join(rootpath, name + ".dat")
        self.maskPath = os.path.join(rootpath, name + "_M.png")
        self.alignmentPath = os.path.join(rootpath, self.imdirPath + "/allignment.txt")

    def __get_alignmentData__(self, scale: float = 1.):
        with open(self.alignmentPath) as file:
            alignDict = json.load(file)
        for k in alignDict.keys():
            alignDict[k]*=scale
        return alignDict


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
        diffImage = (diffImage*0.65 + 128.).astype(int)
        return diffImage

    def __get_metadata__(self, scale: float = 1.):
        return read_metadataFile(self.metadataPath,
                                 scale=scale)

    def __get_anomalousMask__(self, scale: float = 1.):
        """
        Read binary anomaly mask from file.
        """
        if not os.path.isfile(self.maskPath):   
            return None
        else:
            mask = read_singleImage(self.maskPath, scale = scale)
            mask[mask<128] = 0
            mask[mask>=128] = 255
            return mask
        

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


    def __get_allignedRegionMask__(self, region_mask_path, scale = 1.):
        region_mask = read_singleImage(region_mask_path, scale = scale)
        region_mask[region_mask<128] = 0
        region_mask[region_mask>=128] = 1

        # transform the region mask according to alignment info
        align = self.__get_alignmentData__(scale = scale)
        M = np.float32([[1, 0, -align["x"]], [0, 1, -align["y"]]])
        region_mask = cv.warpAffine(region_mask, M, region_mask.shape[::-1])

        return region_mask


    def image_mode_selector(self, mode, scale, minuend, subtrahend):
        """
        Function used inside fetch-functions to select the correct
        image, according to "mode".
        """

        if mode == "diff":
            im_channel = self.__get_diffImage__(scale = scale, minuend = minuend, subtrahend = subtrahend)
            image = np.stack((im_channel, im_channel, im_channel), axis=2)
        elif mode in ["0", "1", "2", "3", "4"]:
            im_channel = self.__get_images__(scale = scale)[int(mode)]
            image = np.stack((im_channel, im_channel, im_channel), axis=2)
        elif mode == "custom_mix":
            #single_lights = self.__get_images__(scale = scale)
            diff_im1 = self.__get_diffImage__(scale = scale, minuend = 3, subtrahend = 0)
            diff_im2 = self.__get_diffImage__(scale = scale, minuend = 2, subtrahend = 1)
            image = np.stack((diff_im1, diff_im1, diff_im2), axis=2)
        else:
            raise(ValueError("Invalid argument: mode"))

        return image


    def fetch_goodCrops(self,
                        N,
                        scale = 1.,
                        size = 224,
                        rand_flip = False,
                        normalize = True,
                        gauss_blur = None,
                        mode = "diff",
                        minuend = 3,
                        subtrahend = 2,
                        region_mask_path = None
                        ):
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
        image = self.image_mode_selector(mode, scale, minuend, subtrahend)
        
        # crops coordinates
        mask = self.__get_goodMask__(scale = scale, size = size)
        # regions restriction
        if region_mask_path is not None:
            # get region mask (with correct allignment)
            region_mask = self.__get_allignedRegionMask__(region_mask_path, scale=scale)
            # combine good mask with the regions restrictions
            mask = mask*region_mask


        centers = self.__get_randomCenters__(mask, N = N)

        # image for binary masks
        mask = self.__get_anomalousMask__(scale = scale)
        if mask is None: # create a zeros mask if mask file does not exists
            mask = np.zeros_like(image) 
        image = np.concatenate((image, np.expand_dims(mask, axis=2)), axis = 2)

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
                             mode = "diff",
                             minuend = 3,
                             subtrahend = 2,
                             min_defect_area = -1,
                             region_mask_path = None
                             ):
        """
        Create a set of anomalous crops using the coordinates from the metadata file.
        (This method is based on the cropImage() function).

        Parameters
        ----------


        Returns
        ----------

        """

        # crops coordinates
        centers, _ = self.__get_metadata__(scale = scale)
        centers = centers[centers[:,4]<1] 

        # regions restriction
        if region_mask_path is not None:
            # get region mask (with correct allignment)
            region_mask = self.__get_allignedRegionMask__(region_mask_path, scale=scale)
            centers = [c for c in centers[:,:2] if region_mask[c[1], c[0]]==1]


        if len(centers)>0:
            # images
            image = self.image_mode_selector(mode, scale, minuend, subtrahend)


            # image for binary masks
            mask = self.__get_anomalousMask__(scale = scale)
            if mask is None: # create a zeros mask if mask file does not exists
                mask = np.zeros_like(image)
            image = np.concatenate((image, np.expand_dims(mask, axis=2)), axis = 2)

            # run cropImage()
            crops, centers = cropImage(image, centers, size = size,
                                    rand_flip = rand_flip,
                                    rand_shift = rand_shift,
                                    normalize = normalize,
                                    gauss_blur = gauss_blur,
                                    min_area = min_defect_area)

        else:
            crops = []
            centers = []

        return crops, centers
    


    def fetch_Bott(self,
                   coords_file,
                   scale = 1.,
                   size = 600,
                   rand_flip = False,
                   normalize = True,
                   gauss_blur = None,
                   mode = "diff",
                   minuend = 3,
                   subtrahend = 2,
                   align = True
                   ):
        """
        Funzione per crop bottoni.

        Parameters
        ----------
        N: number of crops to be fetched.

        Returns
        ----------

        """

        # images
        image = self.image_mode_selector(mode, scale, minuend, subtrahend)
        
        # crops coordinates
        centers = []
        with open(coords_file, 'r') as file:
            for l in file.readlines():
                cx, cy = l.split("\t")
                cx, cy = int(cx.strip()), int(cy.strip())
                centers.append([cx, cy, -1, -1, -1])
        centers = np.asarray(centers)

        if align:
            alignment = self.__get_alignmentData__(scale = scale)
            centers[:,0] -= int(alignment["x"])
            centers[:,1] -= int(alignment["y"])

        # image for binary masks
        mask = self.__get_anomalousMask__(scale = scale)
        if mask is None: # create a zeros mask if mask file does not exists
            mask = np.zeros_like(image) 
        image = np.concatenate((image, np.expand_dims(mask, axis=2)), axis = 2)

        # run cropImage()
        crops, centers = cropImage(image, centers, size = size,
                                   rand_flip = rand_flip,
                                   rand_shift = False,
                                   normalize = normalize,
                                   gauss_blur = gauss_blur)

        return crops, centers
