from glob import glob
from skimage import io
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
import os
import shutil
import json
from tqdm import tqdm

import timm
import torch
import torch.nn as nn

from skimage import data, io
from matplotlib import pyplot as plt


def show(img):
    io.imshow(img)
    plt.show()


device = torch.device("cuda")
encoder = timm.create_model("tf_efficientnet_b5_ns",
                            pretrained=True,
                            features_only=True).to(device)
model = nn.Sequential(nn.ReLU(), nn.Conv2d(176, 3, 1),
                      nn.ReLU()).to(device)  # 112 for B0


def generate_features(image_path, output_dir):
    feature_file_name = f"{output_dir}/{os.path.basename(os.path.dirname(image_path))}_{os.path.basename(image_path).replace('.png','')}.pt"
    image = io.imread(image_path)

    sz = 256

    image = torch.from_numpy(image / 255.0).permute(2, 0, 1).unsqueeze(0)
    images = image.unfold(2, sz, sz).unfold(3, sz, sz)  # .permute(0,2,3,1,4,5)
    images = rearrange(images, "b c p1 p2 w h -> (b p1 p2) c w h")

    features4k = []
    step = 8
    for batch_sz in range(0, images.shape[0], step):  # 4 is batch size here
        batch = images[batch_sz:batch_sz + step]

        features = encoder(batch.float().to(device))
        features = model(features[-2])  #.squeeze()

        features4k.append(features.detach().cpu())

    features4k = torch.vstack(features4k).reshape((16, 16, 3, 16, 16))
    features4k = rearrange(features4k, "b1 b2 c w h -> c (b1 w) (b2 h)")

    torch.save(features4k, feature_file_name)


def init_feature_dir(output_dir):
    try:
        os.mkdir(output_dir)
    except:
        shutil.rmtree(output_dir)
        os.mkdir(output_dir)


dataset_name = "tcga_brca"
with open('configs.json') as json_file:
    configs = json.load(json_file)

images4k_dir = configs['dataset_metadata'][dataset_name]['patches_dir']
images4k_paths = glob(f"{images4k_dir}/*/*.png")

features4k_dir = configs['dataset_metadata'][dataset_name]['features_dir']
init_feature_dir(features4k_dir)

for image_path in tqdm(images4k_paths, total=len(images4k_paths)):
    generate_features(image_path, features4k_dir)
