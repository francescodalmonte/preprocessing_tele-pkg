import os
import numpy as np


def conditionalMkDir(path: str) -> None:
    """Creates a dir if not already existing.
    """
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        os.mkdir(path)



def mkDirTreeFCDD(path: str) -> None:
    """Creates an FCDD-compatible dirtree.
    """
    path = os.path.abspath(path)
    conditionalMkDir(os.path.join(path, "custom"))
    conditionalMkDir(os.path.join(path, "custom/test"))
    conditionalMkDir(os.path.join(path, "custom/train"))
    conditionalMkDir(os.path.join(path, "custom/test/tele"))
    conditionalMkDir(os.path.join(path, "custom/train/tele"))
    conditionalMkDir(os.path.join(path, "custom/test/tele/normal"))
    conditionalMkDir(os.path.join(path, "custom/test/tele/anomalous"))
    conditionalMkDir(os.path.join(path, "custom/train/tele/normal"))
    conditionalMkDir(os.path.join(path, "custom/train/tele/anomalous"))



def randomSplit(path: str, p_good: float = 0., p_anom: float = 0. ) -> None:
    """Randomly split a dataset of nominative and anomalous crops into train and test
    sets (requires a FCDD-compatible dirtree- i.e. custom/train/... , custom/test/... .
    Crops should be initially placed in the train directories).

    Parameters
    ----------
    path: path to root dir of a FCDD-compatible dirtree.
    p_good: fraction of good crops to move in the test set.
    p_anomalous: fraction of good crops to move in the test set.

    """
    path = os.path.abspath(path)

    goodTrainpath = os.path.join(path, "custom/train/tele/normal")
    goodTestpath = os.path.join(path, "custom/test/tele/normal")
    anomTrainpath = os.path.join(path, "custom/train/tele/anomalous")
    anomTestpath = os.path.join(path, "custom/test/tele/anomalous")

    # lists of crops abspaths

    good_crops = [ f for f in os.listdir(goodTrainpath) if f.endswith(".png") ]
    anom_crops = [ f for f in os.listdir(anomTrainpath) if f.endswith(".png") ]

    # random split

    n_good = int(len(good_crops)*p_good)
    n_anom = int(len(anom_crops)*p_anom)
    good_crops_C = np.random.choice(good_crops, n_good, replace = False)
    anom_crops_C = np.random.choice(anom_crops, n_anom, replace = False)

    # move files

    for f in good_crops_C:
        source = os.path.join(goodTrainpath, f)
        destination = os.path.join(goodTestpath, f)
        os.rename(source, destination)
    for f in anom_crops_C:
        source = os.path.join(anomTrainpath, f)
        destination = os.path.join(anomTestpath, f)
        os.rename(source, destination)

