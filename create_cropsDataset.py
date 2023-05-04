import numpy as np
import os
import time

from utils.multiChannelImage import multiChannelImage
from utils.IO import listRawDir
np.random.seed(999)

ROOT = os.path.abspath("C:/Users/Francesco/Pictures/tele/raw")

if __name__ == "__main__":
    start = time.time()

    rawNames = listRawDir(ROOT)
    
    for name in rawNames:
        object = multiChannelImage(name, ROOT)

        anomalousCrops, anomalousCenters = object.fetch_anomalousCrops(scale = 1.,
                                                                       rand_flip = True,
                                                                       rand_shift = True)

        goodCrops, goodCenters = object.fetch_goodCrops(scale = 1., N = 1000)        
        
        print(f"{name} - {len(anomalousCrops)}/{len(goodCrops)}")

    ##### code ####

    print(f"Elapsed time: {(time.time()-start):2f} s")
