####################################################################
###   NB: this code requires the installation of the FCDD framework!
###
####################################################################

import numpy as np
import os
from matplotlib import pyplot as plt

from utils.multiChannelImage import multiChannelImage
from utils.image_processing import tileImage, saveCrops


# TILE INPUT IMAGE

name = "F00000042.Presley-Jetblack_D20210908-160156"
path = "C:/Users/Francesco/Pictures/tele/raw"

object = multiChannelImage(name, path)

image = object.__get_diffImage__()
empty_mask = np.zeros_like(image)
image = np.stack((image, empty_mask), axis = 2)

tiles, grid = tileImage(image, 224, 0, gauss_blur = .8)

# save to file

save_path = os.path.abspath("C:/Users/Francesco/Pictures/tele/inference_test/")
saveCrops(os.path.join(save_path, "tiles"),
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
snapshot = "D:/Users/Francesco/phd/FCDD results/supervised (with masks)/300 epochs/normal_0/it_0/snapshot.pt"

# load model
model = FCDD_CNN224_VGG_F((3,224,224), bias = True)
model.load_state_dict(torch.load(snapshot, map_location=torch.device('cpu'))["net"])

# data transform (should be the same as for training)
transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor()
#            transforms.Normalize([0.,0.,0.], [1.,1.,1.])
            ])

logger = None
quantile = 0.97

# create a dataloader and a trainer
ds = ImageFolder(save_path, transform)
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


# PLOTS

# pick some random samples
fig, ax = plt.subplots(nrows=4, ncols=10, figsize=(14, 9), tight_layout=True)

for i, a in enumerate(ax.reshape(-1)):
    r = np.random.randint(0, len(reduced_ascores))
    
    a.imshow(all_inputs[r, :, :, :].detach().numpy().transpose(1,2,0))
    a.axis("off")
    a.set_title(f"{reduced_ascores[r].detach().numpy():.3f}")

plt.show()