# multiChannelImage class

import os 
import numpy as np

from .IO import read_dirImage, read_metadataFile
from .image_processing import randomFlip, randomShift, cropImage

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
        image_shape = [int(12500*scale), int(4096*scale)]
        mask = np.ones(image_shape)

        centers = self.__get_metadata__(scale = scale)
        if centers is not None:
            for c in centers:
                x, y, w, h, _ = c.astype(int)
                l = int(size//2)
                mask[y-h//2-l : y+h//2+l, x-w//2-l : x+w//2+l] = 0

        return mask


    
    def __get_randomCenters__(self, mask, N):
        """
        Generate random centers coordinates given a binary 2D mask.
        (This function is used to generate the set of good crops).

        """        
        all_coords = np.argwhere(mask)
        idxs =  np.random.randint(0, len(all_coords), size=N)
        
        centers = all_coords[idxs]

        return centers



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



    def fetch_anomalousCrops(self, scale = 1., size = 224,
                             rand_shift = False, rand_flip = False):
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

        crops = cropImage(image, centers, size = size,
                          rand_flip = rand_flip, rand_shift = rand_shift)

        return crops