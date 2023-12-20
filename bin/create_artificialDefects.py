import os
import numpy as np
import cv2 as cv
import shutil

from preprocessing_tele.augmentation import blend

#
# use this script to add N artificially created anomalies to an existing dataset
#
#
#


AUG_PATH = "D:/data/augmentation"
DS_PATH = "D:/data/diff_rad_unsupervised_1k-01k_size224_blur06_marea150_augmented/custom/train/tele/normal"
SAVE_TO_DEFECTS = "D:/data/diff_rad_unsupervised_1k-01k_size224_blur06_marea150_augmented/custom/train/tele/anomalous"
SAVE_TO_MAPS = "D:/data/diff_rad_unsupervised_1k-01k_size224_blur06_marea150_augmented/custom/train_maps/tele/anomalous"
N = 50
ALPHA = .6


if __name__ == "__main__":

    rindexes_defects = np.random.randint(0, len(os.listdir(AUG_PATH))/2, N)
    rindexes_images = np.random.randint(0, len(os.listdir(DS_PATH)), N)

    images_list = [os.path.join(DS_PATH, f) for f in os.listdir(DS_PATH)]

    for i in range(N):
        blended = blend(os.path.join(AUG_PATH, f"{rindexes_defects[i]}.png"),
                        os.path.join(AUG_PATH, f"{rindexes_defects[i]}_M.png"),
                        images_list[rindexes_images[i]],
                        ALPHA)
        
        os.remove(images_list[rindexes_images[i]])
        cv.imwrite(os.path.join(SAVE_TO_DEFECTS, f"{i}.png"), blended)
        shutil.copy(os.path.join(AUG_PATH, f"{rindexes_defects[i]}_M.png"), os.path.join(SAVE_TO_MAPS, f"{i}.png"))
