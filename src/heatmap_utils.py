import requests
import sys
import os

# LinAlg / Stats / Plotting Dependencies
import cv2
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from scipy.stats import rankdata
import skimage.io
from skimage.measure import find_contours
from tqdm import tqdm
import webdataset as wds

# Torch Dependencies
import torch
import torch.nn as nn
import torchvision
import torch.multiprocessing
from torchvision import transforms
from einops import rearrange, repeat
# from HIPT.HIPT_4K.attention_visualization_utils import *


def cmap_map(function, cmap):
    r""" 
    Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    
    Args:
    - function (function)
    - cmap (matplotlib.colormap)
    
    Returns:
    - matplotlib.colormap
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step: np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red', 'green', 'blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j, i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap', cdict, 1024)


################################################
# 256 x 256 ("Patch") Attention Heatmap Creation
################################################
def create_patch_heatmaps_indiv(patch,
                                model256,
                                output_dir,
                                fname,
                                threshold=0.5,
                                offset=16,
                                alpha=0.5,
                                cmap=plt.get_cmap('coolwarm'),
                                device256=torch.device('cuda:0')):
    r"""
    Creates patch heatmaps (saved individually)
    
    To be refactored!

    Args:
    - patch (PIL.Image):        256 x 256 Image 
    - model256 (torch.nn):      256-Level ViT 
    - output_dir (str):         Save directory / subdirectory
    - fname (str):              Naming structure of files
    - offset (int):             How much to offset (from top-left corner with zero-padding) the region by for blending 
    - alpha (float):            Image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): Colormap for creating heatmaps
    
    Returns:
    - None
    """
    # patch1 = patch.copy()
    # patch2 = add_margin(patch.crop((16, 16, 512, 512)),
    #                     top=0,
    #                     left=0,
    #                     bottom=16,
    #                     right=16,
    #                     color=(255, 255, 255))

    patch1 = patch.copy()
    patch2 = add_margin(patch.crop((16, 16, 512, 512)),
                        top=0,
                        left=0,
                        bottom=16,
                        right=16,
                        color=(255, 255, 255))  # changed for 512x512

    print("patch1", patch1.size, "patch2", patch2.size)

    b256_1, a256_1 = get_patch_attention_scores(patch1,
                                                model256,
                                                device256=device256)
    b256_1, a256_2 = get_patch_attention_scores(patch2,
                                                model256,
                                                device256=device256)

    save_region = np.array(patch.copy())
    # s = 256
    s = 512  # changed
    offset_2 = offset

    if threshold != None:
        for i in range(6):
            score256_1 = get_scores256(a256_1[:, i, :, :], size=(s, ) * 2)
            score256_2 = get_scores256(a256_2[:, i, :, :], size=(s, ) * 2)
            new_score256_2 = np.zeros_like(score256_2)
            new_score256_2[offset_2:s,
                           offset_2:s] = score256_2[:(s -
                                                      offset_2), :(s -
                                                                   offset_2)]
            overlay256 = np.ones_like(score256_2) * 100
            overlay256[offset_2:s, offset_2:s] += 100
            score256 = (score256_1 + new_score256_2) / overlay256

            mask256 = score256.copy()
            mask256[mask256 < threshold] = 0
            mask256[mask256 > threshold] = 0.95

            color_block256 = (cmap(mask256) * 255)[:, :, :3].astype(np.uint8)
            region256_hm = cv2.addWeighted(color_block256, alpha,
                                           save_region.copy(), 1 - alpha, 0,
                                           save_region.copy())
            region256_hm[mask256 == 0] = 0
            img_inverse = save_region.copy()
            img_inverse[mask256 == 0.95] = 0
            Image.fromarray(region256_hm + img_inverse).save(
                os.path.join(output_dir, '%s_256th[%d].png' % (fname, i)))

    for i in range(6):
        score256_1 = get_scores256(a256_1[:, i, :, :], size=(s, ) * 2)
        score256_2 = get_scores256(a256_2[:, i, :, :], size=(s, ) * 2)
        new_score256_2 = np.zeros_like(score256_2)
        new_score256_2[offset_2:s,
                       offset_2:s] = score256_2[:(s - offset_2), :(s -
                                                                   offset_2)]
        overlay256 = np.ones_like(score256_2) * 100
        overlay256[offset_2:s, offset_2:s] += 100
        score256 = (score256_1 + new_score256_2) / overlay256
        color_block256 = (cmap(score256) * 255)[:, :, :3].astype(np.uint8)
        region256_hm = cv2.addWeighted(color_block256, alpha,
                                       save_region.copy(), 1 - alpha, 0,
                                       save_region.copy())
        Image.fromarray(region256_hm).save(
            os.path.join(output_dir, '%s_256[%s].png' % (fname, i)))


def get_patch_attention_scores(patch,
                               model256,
                               scale=1,
                               device256=torch.device('cuda:0')):
    r"""
    Forward pass in ViT-256 model with attention scores saved.
    
    Args:
    - region (PIL.Image):       4096 x 4096 Image 
    - model256 (torch.nn):      256-Level ViT 
    - scale (int):              How much to scale the output image by (e.g. - scale=4 will resize images to be 1024 x 1024.)
    
    Returns:
    - attention_256 (torch.Tensor): [1, 256/scale, 256/scale, 3] torch.Tensor of attention maps for 256-sized patches.
    """
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    with torch.no_grad():
        batch_256 = t(patch).unsqueeze(0)
        batch_256 = batch_256.to(device256, non_blocking=True)
        # features_256 = model256(batch_256)
        features_256 = model256(batch_256, features_only=True)
        features_256 = features_256[
            -1]  # Extracting only last efficentNet output

        attention_256 = model256.get_last_selfattention(batch_256)
        print("Attention 256", attention_256.size())
        nh = attention_256.shape[1]  # number of head
        # attention_256 = attention_256[:, :, 0, 1:].reshape(256, nh, -1)
        attention_256 = attention_256[:, :, 0, 1:].reshape(
            1024, nh, -1)  # changed for 512x512 images

        print("attention_256", attention_256.size())

        # attention_256 = attention_256.reshape(1, nh, 16, 16)
        attention_256 = attention_256.reshape(1, nh, 32,
                                              32)  # changed for 512x512

        attention_256 = nn.functional.interpolate(
            attention_256, scale_factor=int(16 / scale),
            mode="nearest").cpu().numpy()

        if scale != 1:
            batch_256 = nn.functional.interpolate(batch_256,
                                                  scale_factor=(1 / scale),
                                                  mode="nearest")

    print("Iteration Complete")
    return tensorbatch2im(batch_256), attention_256


def add_margin(pil_img, top, right, bottom, left, color):
    r"""
    Adds custom margin to PIL.Image.
    """
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def get_scores256(attns, size=(256, 256)):
    r"""
    """
    rank = lambda v: rankdata(v) * 100 / len(v)
    color_block = [rank(attn.flatten()).reshape(size) for attn in attns][0]
    return color_block


def tensorbatch2im(input_image, imtype=np.uint8):
    r""""
    Converts a Tensor array into a numpy image array.
    
    Args:
        - input_image (torch.Tensor): (B, C, W, H) Torch Tensor.
        - imtype (type): the desired type of the converted numpy array
        
    Returns:
        - image_numpy (np.array): (B, W, H, C) Numpy Array.
    """
    if not isinstance(input_image, np.ndarray):
        image_numpy = input_image.cpu().float().numpy(
        )  # convert it into a numpy array
        #if image_numpy.shape[0] == 1:  # grayscale to RGB
        #    image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1)) + 1
                       ) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)
