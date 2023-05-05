import numpy as np
import os
import time

from utils.multiChannelImage import multiChannelImage
from utils.IO import listRawDir
from utils.image_processing import saveCrops
from utils.dataset import mkDirTreeFCDD, randomSplit


DATASET_NAME = "new_FCDD_dataset_23"
N_GOOD = 250
SCALE = 1.
P_GOOD = 0.2
P_ANOM = 1.

SOURCE_ROOT = os.path.abspath("C:/Users/Francesco/Pictures/tele/raw")
SAVE_ROOT = os.path.abspath(f"C:/Users/Francesco/progetti/preprocessing_tele/output/{DATASET_NAME}")

SEED = 999


if __name__ == "__main__":

    start = time.time()
    np.random.seed(SEED)

    rawNames = listRawDir(SOURCE_ROOT)
    mkDirTreeFCDD(SAVE_ROOT)    
    
    print(f"Loading raw images from {SOURCE_ROOT}")

    for name in rawNames:
        object = multiChannelImage(name, SOURCE_ROOT)

        print(f"{name}")

        # extract crops
        anomalousCrops, anomalousCenters = object.fetch_anomalousCrops(scale = SCALE, rand_flip = True, rand_shift = True)
        goodCrops, goodCenters = object.fetch_goodCrops(scale = SCALE, N = N_GOOD)

        print(f"N. anomalous/N. normal: {len(anomalousCrops)}/{len(goodCrops)}")


        # save to file
        os.listdir()

        saveCrops(os.path.join(SAVE_ROOT, "custom/train/tele/anomalous"),
                  anomalousCrops, anomalousCenters, prefix=name+"_")
        saveCrops(os.path.join(SAVE_ROOT, "custom/train/tele/normal"),
                  goodCrops, goodCenters, prefix=name+"_") 


    # split into train and test sets
    randomSplit(SAVE_ROOT, p_good = P_GOOD, p_anom = P_ANOM)

    print(f"Elapsed time: {(time.time()-start):2f} s")
