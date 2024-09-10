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
    conditionalMkDir(path)
    conditionalMkDir(os.path.join(path, "custom"))
    conditionalMkDir(os.path.join(path, "custom/test"))
    conditionalMkDir(os.path.join(path, "custom/train"))
    conditionalMkDir(os.path.join(path, "custom/test_maps"))
    conditionalMkDir(os.path.join(path, "custom/train_maps"))
    conditionalMkDir(os.path.join(path, "custom/test/tele"))
    conditionalMkDir(os.path.join(path, "custom/train/tele"))
    conditionalMkDir(os.path.join(path, "custom/test_maps/tele"))
    conditionalMkDir(os.path.join(path, "custom/train_maps/tele"))
    conditionalMkDir(os.path.join(path, "custom/test/tele/normal"))
    conditionalMkDir(os.path.join(path, "custom/test/tele/anomalous"))
    conditionalMkDir(os.path.join(path, "custom/train/tele/normal"))
    conditionalMkDir(os.path.join(path, "custom/train/tele/anomalous"))
    conditionalMkDir(os.path.join(path, "custom/test_maps/tele/normal"))
    conditionalMkDir(os.path.join(path, "custom/test_maps/tele/anomalous"))
    conditionalMkDir(os.path.join(path, "custom/train_maps/tele/normal"))
    conditionalMkDir(os.path.join(path, "custom/train_maps/tele/anomalous"))



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

    gTrainpath = os.path.join(path, "custom/train/tele/normal")
    gTestpath = os.path.join(path, "custom/test/tele/normal")
    aTrainpath = os.path.join(path, "custom/train/tele/anomalous")
    aTestpath = os.path.join(path, "custom/test/tele/anomalous")

    gTrainpath_M = os.path.join(path, "custom/train_maps/tele/normal")
    gTestpath_M = os.path.join(path, "custom/test_maps/tele/normal")
    aTrainpath_M = os.path.join(path, "custom/train_maps/tele/anomalous")
    aTestpath_M = os.path.join(path, "custom/test_maps/tele/anomalous")

    # lists of crops abspaths

    good_crops = [ f for f in os.listdir(gTrainpath) if f.endswith(".png") ]
    anom_crops = [ f for f in os.listdir(aTrainpath) if f.endswith(".png") ]

    # random split

    n_good = int(len(good_crops)*p_good)
    n_anom = int(len(anom_crops)*p_anom)
    good_crops_C = np.random.choice(good_crops, n_good, replace = False)
    anom_crops_C = np.random.choice(anom_crops, n_anom, replace = False)

    # move files

    for f in good_crops_C:
        source = os.path.join(gTrainpath, f)
        destination = os.path.join(gTestpath, f)
        os.rename(source, destination)

        if os.path.isfile(os.path.join(gTrainpath_M, f)):
            source = os.path.join(gTrainpath_M, f)
            destination = os.path.join(gTestpath_M, f)
            os.rename(source, destination)


    for f in anom_crops_C:
        source = os.path.join(aTrainpath, f)
        destination = os.path.join(aTestpath, f)
        os.rename(source, destination)

        if os.path.isfile(os.path.join(aTrainpath_M, f)):
            source = os.path.join(aTrainpath_M, f)
            destination = os.path.join(aTestpath_M, f)
            os.rename(source, destination)



def randomSplit_byImage(path: str, test_imgs_names: list[str, ]) -> None:
    """Randomly split a dataset of nominative and anomalous crops into train and test
    sets (requires a FCDD-compatible dirtree- i.e. custom/train/... , custom/test/... .
    Crops should be initially placed in the train directories. Crops names are supposed
    to be in the form: "OggettoTelaIntero_F00000001.0_Nnnn_Dyyyymmdd-hhmmss_C(x-y).png").

    Parameters
    ----------
    path: path to root dir of a FCDD-compatible dirtree.
    test_imgs_names: list of images belonging to test set.

    """
    path = os.path.abspath(path)

    print(f"test_imgs_names: {test_imgs_names}")

    gTrainpath = os.path.join(path, "custom/train/tele/normal")
    gTestpath = os.path.join(path, "custom/test/tele/normal")
    aTrainpath = os.path.join(path, "custom/train/tele/anomalous")
    aTestpath = os.path.join(path, "custom/test/tele/anomalous")

    gTrainpath_M = os.path.join(path, "custom/train_maps/tele/normal")
    gTestpath_M = os.path.join(path, "custom/test_maps/tele/normal")
    aTrainpath_M = os.path.join(path, "custom/train_maps/tele/anomalous")
    aTestpath_M = os.path.join(path, "custom/test_maps/tele/anomalous")

    # lists of crops abspaths

    good_crops = [ f for f in os.listdir(gTrainpath) if (f.endswith(".jpg") or f.endswith(".png")) ]
    anom_crops = [ f for f in os.listdir(aTrainpath) if (f.endswith(".jpg") or f.endswith(".png")) ]

    # lists of test set crops

    good_crops_C = [x for x in good_crops if (os.path.split(x)[1]).rsplit("_", 1)[0] in test_imgs_names]
    anom_crops_C = [x for x in anom_crops if (os.path.split(x)[1]).rsplit("_", 1)[0] in test_imgs_names]

    print(len(good_crops_C), len(anom_crops_C))

    # move files

    for f in good_crops_C:
        source = os.path.join(gTrainpath, f)
        destination = os.path.join(gTestpath, f)
        os.rename(source, destination)

        if os.path.isfile(os.path.join(gTrainpath_M, f)):
            source = os.path.join(gTrainpath_M, f)
            destination = os.path.join(gTestpath_M, f)
            os.rename(source, destination)


    for f in anom_crops_C:
        source = os.path.join(aTrainpath, f)
        destination = os.path.join(aTestpath, f)
        os.rename(source, destination)

        if os.path.isfile(os.path.join(aTrainpath_M, f)):
            source = os.path.join(aTrainpath_M, f)
            destination = os.path.join(aTestpath_M, f)
            os.rename(source, destination)
