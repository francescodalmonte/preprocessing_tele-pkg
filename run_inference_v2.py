########################################################################
#####                                                              #####
#####     NB: This code requires the FCDD framework installed!     #####
#####                                                              #####
########################################################################

import os
import numpy as np
from matplotlib import pyplot as plt
from typing import Dict, List

import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torchvision.transforms as transforms


from utils.multiChannelImage import multiChannelImage
from utils.image_processing import tileImage

from fcdd.models.fcdd_cnn_224 import FCDD_CNN224_VGG_F
from fcdd.training.fcdd import FCDDTrainer


config = {"INPUT_NAME": "F00000042.Presley-Jetblack_D20210908-160156",
          "INPUT_PATH": "C:/Users/Francesco/Pictures/tele/raw",
          "SNAPSHOT": "D:/Users/Francesco/phd/FCDD results/supervised (with masks)/10 epochs/normal_0/it_0/snapshot.pt",
          "SAVE_PATH": "C:/Users/Francesco/Pictures/tele/inference_results/",
          "SIZE": 224,
          "OVERLAP": 0,
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

    


if __name__ == "__main__":
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


        # forward 

        inputs_set = []
        coords_set = []
        anomaly_maps_set = []
        anomaly_scores_set = []

        for i, (input, y, coords) in enumerate(dl):
            if i < 100:
                if i%10==0: print(f"processing {i}/{len(dl)}")
                with torch.no_grad():
                    out = trainer.net(input.float())
                    anomaly_map = trainer.anomaly_score(trainer.loss(out, input, y, reduce='none'))
                    anomaly_map = trainer.net.receptive_upsample(anomaly_map, reception=True, std=8, cpu=False)
                    anomaly_score = trainer.reduce_ascore(anomaly_map)

                    inputs_set.append(input.detach().numpy()[0,0,:,:])
                    coords_set.append(coords.detach().numpy())
                    anomaly_maps_set.append(anomaly_map.detach().numpy()[0,0,:,:])
                    anomaly_scores_set.append(anomaly_score.detach().numpy()[0])

        inputs_set = np.array(inputs_set)
        coords_set = np.array(coords_set)
        anomaly_maps_set = np.array(anomaly_maps_set)
        anomaly_scores_set = np.array(anomaly_scores_set)

        print(inputs_set.shape)
        print(coords_set.shape)
        print(anomaly_maps_set.shape)
        print(anomaly_scores_set.shape)

