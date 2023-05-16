import numpy as np
import os
import time
import configparser

from utils.multiChannelImage import multiChannelImage
from utils.IO import listRawDir
from utils.image_processing import saveCrops
from utils.dataset import mkDirTreeFCDD, randomSplit


def setupArgs():
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), "create_cropsDataset.INI")
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

        # extract crops
        anomalousCrops, anomalousCenters = object.fetch_anomalousCrops(scale = float(config['SCALE']),
                                                                       rand_flip = True,
                                                                       rand_shift = True,
                                                                       normalize = True,
                                                                       gauss_blur = .8
                                                                       )
        goodCrops, goodCenters = object.fetch_goodCrops(scale = float(config['SCALE']),
                                                        N = int(config['N_GOOD']),
                                                        rand_flip = True,
                                                        normalize = True,
                                                        gauss_blur = .8
                                                        )

        print(f"N. anomalous/N. normal: {len(anomalousCrops)}/{len(goodCrops)}")

        # save to file
        saveCrops(os.path.join(config['SAVE_ROOT'], "custom/train/tele/normal"),
                  goodCrops[:,:,:,0],
                  goodCenters,
                  prefix = name+"_"
                  ) 

        if len(anomalousCrops>0):
            saveCrops(os.path.join(config['SAVE_ROOT'], "custom/train/tele/anomalous"),
                      anomalousCrops[:,:,:,0],
                      anomalousCenters,
                      prefix = name+"_"
                      )
            saveCrops(os.path.join(config['SAVE_ROOT'], "custom/train_maps/tele/anomalous"),
                      anomalousCrops[:,:,:,1],
                      anomalousCenters,
                      prefix = name+"_"
                      )
        

    # split into train and test sets
    randomSplit(config['SAVE_ROOT'],
                p_good = float(config['P_GOOD']),
                p_anom = float(config['P_ANOM']))

    print(f"Elapsed time: {(time.time()-start):2f} s")
