import numpy as np
import os
import time
import configparser

from utils.multiChannelImage import multiChannelImage
from utils.IO import listRawDir
from utils.image_processing import saveCrops, cropImage
from utils.dataset import mkDirTreeFCDD


def setupArgs():
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), "create_cropsDs_Multimodal.INI")
    if os.path.isfile(config_path):
        config.read(config_path)
    else:
        raise ValueError(f"can't find configuration file {config_path}")
    
    return config


if __name__ == "__main__":

    # setup input arguments
    config = setupArgs()["DEFAULT"]

    start = time.time()
    np.random.seed(int(config['SEED']))

    rawNames = listRawDir(config['SOURCE_ROOT'])
    mkDirTreeFCDD(config['SAVE_ROOT'])    
    
    print(f"Loading raw images from {config['SOURCE_ROOT']}")

    for name in rawNames:
        object = multiChannelImage(name, config['SOURCE_ROOT'])

        print(f"{name}")

        # centers
        mask = object.__get_goodMask__(scale = float(config["SCALE"]), size = 224)
        centers = object.__get_randomCenters__(mask, N = int(config["N_GOOD"]))        

        crops_sets = []
        centers_sets = []
        # image
        for image in [object.__get_diffImage__(scale = float(config["SCALE"])),
                      object.__get_images__(scale = float(config["SCALE"]))[2],
                      object.__get_images__(scale = float(config["SCALE"]))[3]]:
            

            # run cropImage()
            crops, centers = cropImage(image,
                                       centers,
                                       size = 224,
                                       rand_flip = False,
                                       rand_shift = False,
                                       normalize = False,
                                       gauss_blur = 0.8)
            
            crops_sets.append(crops)
            centers_sets.append(centers)

        # save to file
        saveCrops(os.path.join(config['SAVE_ROOT'], "custom/train/tele/normal"),
                  crops_sets[0],
                  centers_sets[0],
                  suffix = "_diff"
                  ) 
        saveCrops(os.path.join(config['SAVE_ROOT'], "custom/train/tele/normal"),
                  crops_sets[1],
                  centers_sets[1],
                  suffix = "_2"
                  ) 
        saveCrops(os.path.join(config['SAVE_ROOT'], "custom/train/tele/normal"),
                  crops_sets[2],
                  centers_sets[2],
                  suffix = "_3"
                  ) 
