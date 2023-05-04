import numpy as np
import time

from preprocess.utils.multiChannelImage import multiChannelImage

np.random.seed(999)

if __name__ == "__main__":
    start = time.time()

    ##### code ####

    print(f"Finish! Elapsed time: {(time.time()-start):2f} s")
