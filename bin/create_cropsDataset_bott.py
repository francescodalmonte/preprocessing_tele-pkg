import numpy as np
import os
import time
import configparser

from preprocessing_tele.multiChannelImage import multiChannelImage
from preprocessing_tele.IO import listRawDir, saveInfo
from preprocessing_tele.image_processing import saveCrops
from preprocessing_tele.dataset import mkDirTreeFCDD, randomSplit, randomSplit_byImage


def setupArgs():
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), "create_cropsDataset_bott.INI")
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
    anomalous_names = [n.strip() for n in config["ANOMALOUS_NAMES"].split(",")]
    print(f"Anomalous imgs: {config['ANOMALOUS_NAMES'].split(',')} ")

    for name in rawNames:
        name = name.strip()
        if name not in anomalous_names:
            mcObject = multiChannelImage(name, config['SOURCE_ROOT'])
            print(f"{name}")

            # extract crops
            crops, centers = mcObject.fetch_Bott(config["BOTT_COORDS_TXTFILE_PATH"],
                                                 scale = float(config['SCALE']),
                                                 size = float(config['SIZE']),
                                                 rand_flip = False,
                                                 normalize = bool(int(config["NORMALIZE_CROPS"])),
                                                 gauss_blur = float(config['GAUSS_BLUR']),
                                                 mode = config['MODE'],
                                                 minuend = int(config['DIFF_MINUEND']),
                                                 subtrahend = int(config['DIFF_SUBTRAHEND']),
                                                 align=True
                                                 )


            # save to file
            saveCrops(os.path.join(config['SAVE_ROOT'], "custom/train/tele/normal"),
                      crops[:,:,:,0],
                      centers,
                      prefix = name+"_"
                      ) 
 
        else:
            mcObject = multiChannelImage(name, config['SOURCE_ROOT'])
            print(f"{name} **anomalous")

            # extract crops
            crops, centers = mcObject.fetch_Bott(config["BOTT_COORDS_TXTFILE_PATH"],
                                                 scale = float(config['SCALE']),
                                                 size = float(config['SIZE']),
                                                 rand_flip = False,
                                                 normalize = bool(int(config["NORMALIZE_CROPS"])),
                                                 gauss_blur = float(config['GAUSS_BLUR']),
                                                 mode = config['MODE'],
                                                 minuend = int(config['DIFF_MINUEND']),
                                                 subtrahend = int(config['DIFF_SUBTRAHEND']),
                                                 align=True
                                                 )
            
            saveCrops(os.path.join(config['SAVE_ROOT'], "custom/train/tele/anomalous"),
                      crops[:,:,:,0],
                      centers,
                      prefix = name+"_"
                      )
            saveCrops(os.path.join(config['SAVE_ROOT'], "custom/train_maps/tele/anomalous"),
                      crops[:,:,:,1],
                      centers,
                      prefix = name+"_"
                      )
        

    # split into train and test sets
    if bool(int(config["TEST_SPLIT_BY_IMAGE"])):
        randomSplit_byImage(config['SAVE_ROOT'], config['TEST_IMAGES_NAMES'])
    else:
        randomSplit(config['SAVE_ROOT'],
                    p_good = float(config['P_GOOD']),
                    p_anom = float(config['P_ANOM']))

    # save .txt info file
    saveInfo(config, config['SAVE_ROOT'])

    print(f"Elapsed time: {(time.time()-start):2f} s")



