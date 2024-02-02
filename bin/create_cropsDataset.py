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

    print(f"Loading raw images from {config['SOURCE_ROOT']}")
    excluded = [n.strip() for n in config["EXCLUDED_NAMES"].split(",")]
    print(f"Excluding images: {excluded}")

    if len(config['REGION_MASK_PATH'])>0:
        region_mask_path = config['REGION_MASK_PATH']
        print(f"Using region mask: {region_mask_path}")
    else: 
        region_mask_path=None

    rawNames = listRawDir(config['SOURCE_ROOT'], excluded)
    mkDirTreeFCDD(config['SAVE_ROOT'])    
    

    for name in rawNames:
        mcObject = multiChannelImage(name, config['SOURCE_ROOT'])

        print(f"{name}")

        # extract crops
        anomalousCrops, anomalousCenters = mcObject.fetch_anomalousCrops(scale = float(config['SCALE']),
                                                                       size = float(config['SIZE']),
                                                                       rand_flip = False,
                                                                       rand_shift = True,
                                                                       normalize = bool(int(config["NORMALIZE_CROPS"])),
                                                                       gauss_blur = float(config['GAUSS_BLUR']),
                                                                       mode = config['MODE'],
                                                                       minuend = int(config['DIFF_MINUEND']),
                                                                       subtrahend = int(config['DIFF_SUBTRAHEND']),
                                                                       min_defect_area = int(config['MIN_DEFECT_AREA']),
                                                                       region_mask_path = region_mask_path
                                                                       )
        goodCrops, goodCenters = mcObject.fetch_goodCrops(scale = float(config['SCALE']),
                                                        size = float(config['SIZE']),
                                                        N = int(config['N_GOOD']),
                                                        rand_flip = False,
                                                        normalize = bool(int(config["NORMALIZE_CROPS"])),
                                                        gauss_blur = float(config['GAUSS_BLUR']),
                                                        mode = config['MODE'],
                                                        minuend = int(config['DIFF_MINUEND']),
                                                        subtrahend = int(config['DIFF_SUBTRAHEND']),
                                                        region_mask_path = region_mask_path                                                     
                                                        )

        print(f"N. anomalous/N. normal: {len(anomalousCrops)}/{len(goodCrops)}")

        # save to file
        saveCrops(os.path.join(config['SAVE_ROOT'], "custom/train/tele/normal"),
                  goodCrops[:,:,:,:3],
                  goodCenters,
                  prefix = name+"_"
                  ) 


        if len(anomalousCrops)>0:
            saveCrops(os.path.join(config['SAVE_ROOT'], "custom/train/tele/anomalous"),
                      anomalousCrops[:,:,:,:3],
                      anomalousCenters,
                      prefix = name+"_"
                      )
            saveCrops(os.path.join(config['SAVE_ROOT'], "custom/train_maps/tele/anomalous"),
                      anomalousCrops[:,:,:,3],
                      anomalousCenters,
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
