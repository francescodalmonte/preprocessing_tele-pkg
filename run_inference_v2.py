########################################################################
#####                                                              #####
#####     NB: This code requires the FCDD framework installed!     #####
#####                                                              #####
########################################################################

import os
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from typing import Dict, List

import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torchvision.transforms as transforms


from utils.multiChannelImage import multiChannelImage
from utils.image_processing import tileImage
from utils.dataset import conditionalMkDir

from fcdd.models.fcdd_cnn_224 import FCDD_CNN224_VGG_F
from fcdd.training.fcdd import FCDDTrainer


config = {"INPUT_NAME": "F00000042.Presley-Jetblack_D20210908-160156",
          "INPUT_PATH": "C:/Users/Francesco/Pictures/tele/raw",
          "SNAPSHOT": "D:/Users/Francesco/phd/FCDD results/supervised (with masks)/10 epochs/normal_0/it_0/snapshot.pt",
          "SAVE_PATH": "C:/Users/Francesco/Pictures/tele/inference_results/",
          "SIZE": 224,
          "OVERLAP": 30,
          "DEVICE": "cpu",
          "NORMALIZE": [0.5, 0.0368] # cpu or cuda
          }



def tile_input_image(name: str,
                     root_path: str,
                     size: int = 224, 
                     overlap: int = 0
                     ):
    object = multiChannelImage(name, root_path)
    image = object.__get_diffImage__()
    image = np.stack((image, np.zeros_like(image)), axis = 2)

    tiles, coords = tileImage(image, size, overlap, gauss_blur = .8)
    tiles = np.stack((tiles, tiles, tiles)).transpose(1,2,3,0)

    return tiles/255, coords, image



def setup_model(snapshot: str,
                device: str, 
                size: int = 224
                ):
    model = FCDD_CNN224_VGG_F((3, size, size),
                              bias = True)
    snapshot = torch.load(snapshot, map_location=torch.device(device))
    model.load_state_dict(snapshot["net"])
    return model

def setup_trainer(model,
                  size
                  ):
    trainer = FCDDTrainer(model, None, None, (None, None), None, 'fcdd', 8, 0.97, size)
    trainer.net.eval()
    return trainer


class CustomDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tiles, coords, transform = None):
        self.tiles = tiles
        self.coords = coords
        self.transform = transform

    def __getitem__(self, index):
        x = self.tiles[index]
        y = 0
        c = self.coords[index]

        if self.transform:
            x = self.transform(x)

        return x, y, c

    def __len__(self):
        return len(self.tiles)



def setup_dataloader(tiles: List[np.ndarray],
                     coords: List[float],
                     M: float = 0.5000,
                     S: float = 0.0368
                     ):
    
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((M,M,M), (S,S,S))
            ])
    
    ds = CustomDataset(tiles, coords, transform)
    dl = DataLoader(ds, batch_size=1, num_workers=0)

    return dl


def save_anomaly_hist(anomaly_scores_set: np.ndarray,
                      save_path: str
                      ):
    d = np.clip(anomaly_scores_set, 0, 1)
    plt.hist(anomaly_scores_set, bins = 50, range = (-0.01, 1.01))
    plt.yscale("log")
    plt.title("Anomaly scores (clipped [0.0:1.0])")
    plt.savefig(save_path)



def save_anomaly_heatmap(coords_set: np.ndarray,
                         anomaly_scores_set: np.ndarray,
                         save_path: str
                        ):
    h = anomaly_scores_set.reshape(len(np.unique(coords_set[:,1])),
                                   len(np.unique(coords_set[:,0])))
        
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,12), dpi=600)
    ax[0].imshow(image[:,:,0]); ax[0].axis("off")
    ax[1].imshow(np.sqrt(h)); ax[1].axis("off")
    
    plt.savefig(save_path)



def save_annotated_image(image: np.ndarray,
                         coords_set: np.ndarray,
                         anomaly_scores_set: np.ndarray,
                         save_path: str,
                         threshold: float = 0.1
                         ):
    # image
    CVimage = np.array(np.stack((image, image, image)).transpose(1,2,0)).copy()

    # threshold mask
    m = anomaly_scores_set > 0.1

    for c, s in zip(coords_set[m], anomaly_scores_set[m]):
        # draw rectangle
        topleft = (int(c[0])-224//2, int(c[1])-224//2)
        bottomright = (int(c[0])+224//2, int(c[1])+224//2)

        # map color according to anomaly score
        r = int(220*np.clip(s, 0, 1))
        g = 0
        b = int(r*0.1)
        w = int(8*np.clip(s,0,1))
        tw = int(3*np.clip(s,0,1))
        color = (b,g,r)

        cv.rectangle(CVimage, topleft, bottomright, color, w)
        cv.putText(CVimage, f"{s:.5f}", (topleft[0]+5, topleft[1]-5), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), tw)

    # save image to file
    cv.imwrite(save_path, CVimage)


if __name__ == "__main__":
        
        # SETUP 

        tiles, coords, image = tile_input_image(config["INPUT_NAME"],
                                                config["INPUT_PATH"],
                                                config["SIZE"],
                                                config["OVERLAP"])
        
        model = setup_model(config["SNAPSHOT"],
                            config["DEVICE"],
                            config["SIZE"])
        
        dl = setup_dataloader(tiles,
                              coords,
                              config["NORMALIZE"][0],
                              config["NORMALIZE"][1])
        
        trainer = setup_trainer(model,
                                config["SIZE"])



        # FORWARD LOOP
        
        inputs_set = []
        coords_set = []
        anomaly_maps_set = []
        anomaly_scores_set = []

        for i, (input, y, coords) in enumerate(dl):
            if i%10==0: print(f"processing {i}/{len(dl)}")
            with torch.no_grad():
                out = trainer.net(input.float())
                anomaly_map = trainer.anomaly_score(trainer.loss(out, input, y, reduce='none'))
                anomaly_map = trainer.net.receptive_upsample(anomaly_map, reception=True, std=8, cpu=False)
                anomaly_score = trainer.reduce_ascore(anomaly_map)

                inputs_set.append(input.detach().numpy()[0,0,:,:])
                coords_set.append(coords.detach().numpy()[0])
                anomaly_maps_set.append(anomaly_map.detach().numpy()[0,0,:,:])
                anomaly_scores_set.append(anomaly_score.detach().numpy()[0])

        inputs_set = np.array(inputs_set)
        coords_set = np.array(coords_set)
        anomaly_maps_set = np.array(anomaly_maps_set)
        anomaly_scores_set = np.array(anomaly_scores_set)


        # SAVE RESULTS

        conditionalMkDir(config["SAVE_PATH"])

        np.save(os.path.join(config["SAVE_PATH"], "inputs_set.npy"), inputs_set)
        np.save(os.path.join(config["SAVE_PATH"], "coords_set.npy"), coords_set)
        np.save(os.path.join(config["SAVE_PATH"], "anomaly_maps_set.npy"), anomaly_maps_set)
        np.save(os.path.join(config["SAVE_PATH"], "anomaly_scores_set.npy"), anomaly_scores_set)
        """
        inputs_set = np.load(os.path.join(config["SAVE_PATH"], "inputs_set.npy"))
        coords_set = np.load(os.path.join(config["SAVE_PATH"], "coords_set.npy"))
        anoamly_maps_set = np.load(os.path.join(config["SAVE_PATH"], "anomaly_maps_set.npy"))
        anomaly_scores_set = np.load(os.path.join(config["SAVE_PATH"], "anomaly_scores_set.npy"))
        """
        
        save_anomaly_hist(anomaly_scores_set,
                          os.path.join(config["SAVE_PATH"], "anomaly_hist.png"))

        save_anomaly_heatmap(coords_set,
                             anomaly_scores_set,
                             os.path.join(config["SAVE_PATH"], "anomaly_heatmap.png"))
        
        save_annotated_image(image[:,:,0],
                             coords_set,
                             anomaly_scores_set,
                             os.path.join(config["SAVE_PATH"], "annotated_image.png"))
        


