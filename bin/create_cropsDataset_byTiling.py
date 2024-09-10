import numpy as np
import os
import time
import configparser

from preprocessing_tele.multiChannelImage import multiChannelImage
from preprocessing_tele.IO import listRawDir, saveInfo
from preprocessing_tele.image_processing import saveCrops, tileImage
from preprocessing_tele.dataset import mkDirTreeFCDD, randomSplit, randomSplit_byImage


def setupArgs():
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), "create_cropsDataset_byTiling.INI")
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

    if len(config['MASK_THRESHOLD'])>0:
        mask_threshold = [int(t.strip()) for t in config["MASK_THRESHOLD"].split(",")]
        print(f"Using mask threshold: {mask_threshold}")
    else:
        mask_threshold = [1, 256]

    if len(config['MAX_LATERAL_DISTANCE'].strip())>0:
        max_lateral_dist = int(config['MAX_LATERAL_DISTANCE'])
    else:
        max_lateral_dist = None

    if len(config['MIN_LATERAL_DISTANCE'].strip())>0:
        min_lateral_dist = int(config['MIN_LATERAL_DISTANCE'])
    else:
        min_lateral_dist = None
    

    rawNames = listRawDir(config['SOURCE_ROOT'], excluded)
    mkDirTreeFCDD(config['SAVE_ROOT'])    
    
    if bool(int(config["ONLY_TEST_DS"])): print("****Running in ONLY_TEST_DS mode****")
    for name in rawNames:
        if bool(int(config["ONLY_TEST_DS"])) and (name not in config['TEST_IMAGES_NAMES']):
            continue
        else:

            mcObject = multiChannelImage(name, config['SOURCE_ROOT'])

            print(f"{name}")

            # extract crops

            image = mcObject.image_mode_selector(mode = config['MODE'],
                                                 scale = float(config['SCALE']),
                                                 term1 = int(config['TERM1']),
                                                 term2 = int(config['TERM2']),
                                                 contrast_correction = bool(int(config['CONTRAST_CORRECTION'])),
                                                 contrast_correction_sigma = float(config['CONTRAST_CORRECTION_SIGMA'])
                                                )
            mask = mcObject.__get_anomalousMask__(scale = float(config['SCALE']))

            # append mask along the color channel
            image = np.dstack((image, mask))#[60:,:1500]
            print(f"image shape: {image.shape} -------> !! CHECK !!")

            crops, centers = tileImage(image, 
                                       size = float(config['SIZE']),
                                       overlap = float(config['OVERLAP']),
                                       normalize = bool(int(config["NORMALIZE_CROPS"])),
                                       gauss_blur = float(config['GAUSS_BLUR']),
                                       min_area = -1,               # at this stage don't filter by mask area
                                       mask_threshold = [0, 256],    # at this stage don't filter by mask threshold
                                       saturate_mask = False)

            # split into normal and anomalous crops
            goodCrops, goodCenters, anomalousCrops, anomalousCenters = [], [], [], []

            for i, crop in enumerate(crops):
                if crop.shape == (float(config['SIZE']), float(config['SIZE']), 4): # crops WITH mask
                    if np.all(crop[:,:,3]==0):
                        goodCrops.append(crop)
                        goodCenters.append(centers[i])

                    else:
                        b = int(config['BORDER'])
                        effective_mask = np.logical_and(crop[b:-b,b:-b,3]>mask_threshold[0], crop[b:-b,b:-b,3]<mask_threshold[1])
                        if np.any(effective_mask): # filter by mask threshold
                            if np.sum(effective_mask) > int(config['MIN_DEFECT_AREA']):  # filter by mask area
                                anomalousCrops.append(crop)
                                anomalousCenters.append(centers[i])
                elif crop.shape == (float(config['SIZE']), float(config['SIZE']), 3):
                    goodCrops.append(crop)
                    goodCenters.append(centers[i])
            
            goodCrops, goodCenters = np.array(goodCrops), np.array(goodCenters)
            anomalousCrops, anomalousCenters = np.array(anomalousCrops), np.array(anomalousCenters)

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
    randomSplit_byImage(config['SAVE_ROOT'], config['TEST_IMAGES_NAMES'])

    # save .txt info file
    saveInfo(config, config['SAVE_ROOT'])

    print(f"Elapsed time: {(time.time()-start):2f} s")
