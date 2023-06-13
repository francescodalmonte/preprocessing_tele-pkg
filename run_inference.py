####################################################################
###   NB: this code requires the installation of the FCDD framework!
###
####################################################################

import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import patches as patches

from utils.dataset import conditionalMkDir
from utils.multiChannelImage import multiChannelImage
from utils.image_processing import tileImage, saveCrops


# TILE INPUT IMAGE

name = "F00000042.Presley-Jetblack_D20210908-160156"
path = "C:/Users/Francesco/Pictures/tele/raw"

object = multiChannelImage(name, path)

image = object.__get_diffImage__()
empty_mask = np.zeros_like(image)
image = np.stack((image, empty_mask), axis = 2)

tiles, grid = tileImage(image, 224, 20, gauss_blur = .8)

# save to file

ds_path = os.path.abspath("C:/Users/Francesco/Pictures/tele/inference_test/")
saveCrops(os.path.join(ds_path, "tiles"),
          tiles[:,:,:,0],
          grid,
          prefix = name+"_"
          ) 


# INFERENCE

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

from fcdd.training.fcdd import FCDDTrainer
from fcdd.datasets.image_folder import ImageFolder
from fcdd.models.fcdd_cnn_224 import FCDD_CNN224_VGG_F

# path to model snapshot
snapshot = "D:/Users/Francesco/phd/FCDD results/supervised (with masks)/10 epochs/normal_0/it_0/snapshot.pt"

# load model
model = FCDD_CNN224_VGG_F((3,224,224), bias = True)
model.load_state_dict(torch.load(snapshot, map_location=torch.device('cpu'))["net"])

# data transform (should be the same as for training)
transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.0368, 0.0368, 0.0368))
            ])

logger = None
quantile = 0.97

# create a dataloader and a trainer
ds = ImageFolder(ds_path, transform, transforms.Lambda(lambda x: 0))
loader = DataLoader(ds, batch_size=1, num_workers=0)

trainer = FCDDTrainer(model, None, None, (None, None), logger, 'fcdd', 8, quantile, 224)
trainer.net.eval()

# forward
all_anomaly_scores, all_inputs, all_labels = [], [], []
for i, (inputs, labels) in enumerate(loader):
    if i%10==0: print(f"processing {i}/{len(loader)}")
    with torch.no_grad():
        outputs = trainer.net(inputs)
        anomaly_scores = trainer.anomaly_score(trainer.loss(outputs, inputs, labels, reduce='none'))
        anomaly_scores = trainer.net.receptive_upsample(anomaly_scores, reception=True, std=8, cpu=False)
        all_anomaly_scores.append(anomaly_scores.cpu())
        all_inputs.append(inputs.cpu())
        all_labels.append(labels)
all_inputs = torch.cat(all_inputs)
all_labels = torch.cat(all_labels)

# reduce anomaly scores (from pixel-wise to sample-wise)
all_anomaly_scores = torch.cat(all_anomaly_scores)
reduced_ascores = trainer.reduce_ascore(all_anomaly_scores)


# RESULTS
save_path = os.path.abspath("C:/Users/Francesco/Pictures/tele/inference_results/")
save_path_results = os.path.join(save_path, name)
conditionalMkDir(save_path_results)

input_names = np.array(sorted(os.listdir(os.path.join(ds_path, "tiles"))))

input_images = all_inputs.detach().numpy().transpose(0,2,3,1)
input_images = np.clip(input_images*25 + 128, 0, 255).astype(int)

output_scores = reduced_ascores.detach().numpy()

# plot anomaly 
plt.hist(np.clip(output_scores, 0, 10), 50, (-1,11))
plt.yscale("log")
plt.title("Anomaly scores")
plt.savefig(os.path.join(save_path_results, "anomaly_scores_hist.png"))

# threshold anomalies
print("input a threshold value (press enter to confirm):")
thresh = float(input())
mask_thresh = output_scores > thresh

print(np.sum(mask_thresh))

Tscores = output_scores[mask_thresh]
Timages = input_images[mask_thresh]
Tnames = input_names[mask_thresh]

# Create figure and draw rectangles
import cv2 as cv

# image
CVimage = np.array(image[:,:,0])

for i, n in enumerate(Tnames):
    # fetch coordinates from file name
    c = np.array(n.split("(")[1].split(")")[0].split("-")).astype(float)
    # draw rectangle
    topleft = (int(c[0])-224//2, int(c[1])-224//2)
    bottomright = (int(c[0])+224//2, int(c[1])+224//2)
    cv.rectangle(CVimage, topleft, bottomright,(0,0,0),3)
    cv.putText(CVimage, f"{Tscores[i]:.5f}", (topleft[0]+5, topleft[1]-5), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# save image to file
cv.imwrite(os.path.join(save_path_results, "image.png"), CVimage)
