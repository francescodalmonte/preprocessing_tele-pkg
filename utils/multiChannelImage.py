# multiChannelImage class

import os 

from IO import read_dirImage, read_metadataFile

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