import json
import os
import numpy as np
import shutil

from time import perf_counter
from tiatoolbox.wsicore.wsireader import WSIReader
from pprint import pprint
from tqdm import tqdm
from glob import glob
from skimage import io

from joblib import Parallel, delayed
from math import sqrt


def create_4k_patches(wsi_path, output_dir):
    try:
        wsi_dir = os.path.basename(wsi_path).replace('.svs', '')
        wsi = WSIReader.open(input_img=wsi_path)
        mask = wsi.tissue_mask(resolution=2, units="level")
        #     wsi_thumb = wsi.slide_thumbnail(resolution=1.25, units="power")
        #     mask_thumb = mask.slide_thumbnail(resolution=1.25, units="power")

        #     show_side_by_side(wsi_thumb, mask_thumb)
        #     continue

        wsi_info = wsi.info.as_dict()

        try:
            os.mkdir(f'{output_dir}/{wsi_dir}')
        except:
            shutil.rmtree(f'{output_dir}/{wsi_dir}')
            os.mkdir(f'{output_dir}/{wsi_dir}')

        wsi20x_h, wsi20x_w = wsi_info['level_dimensions'][1][::-1]
        sz = 4096
        for y in range(0, wsi20x_h // sz):
            for x in range(0, wsi20x_w // sz):

                wsi_region = wsi.read_region(location=(y * sz, x * sz),
                                             level=1,
                                             size=(sz, sz))
                mask_region = mask.read_region(location=(y * sz, x * sz),
                                               level=1,
                                               size=(sz, sz))

                unique, counts = np.unique(mask_region, return_counts=True)
                if (unique[0] == 0 and counts[0] >
                    ((mask_region.shape[0] * mask_region.shape[1]) *
                     0.80)):  #(unique.shape[0] != 2 and unique[0]==0) or
                    continue

                io.imsave(f"{output_dir}/{wsi_dir}/{y}_{x}.png", wsi_region)
    except:
        raise Exception("Failed to Load ", wsi_path)


dataset_name = "tcga_brca"
with open('configs.json') as json_file:
    configs = json.load(json_file)

PARALLEL_RUN = True
brca_wsis = glob(
    f"{configs['dataset_metadata'][dataset_name]['wsi_dir']}/*/*.svs")
output_dir = configs['dataset_metadata'][dataset_name]['patches_dir']

if PARALLEL_RUN:
    Parallel(n_jobs=8)(delayed(create_4k_patches)(wsi_path, output_dir)
                       for wsi_path in tqdm(brca_wsis, total=len(brca_wsis)))
else:
    idx = 0
    for wsi_path in tqdm(brca_wsis, total=len(brca_wsis)):
        create_4k_patches(wsi_path, output_dir)
