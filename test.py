import torch
import torchio as tio
from torch.utils.data import DataLoader

from zipfile import ZipFile
from glob import glob
import os, re
import shutil
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt

from datetime import datetime

from monai.data import ITKReader, PILReader, decollate_batch, CacheDataset

from  monai.transforms import *

from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR, UNet, DynUNet, SegResNet
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference

# Import our utility functions
import models
from dataset import get_datasets

import config

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    _, val_dataset = get_datasets()

    unet = models.load_pretrained_3D_UNET(path="./trained_models/UNET_best_metric_model.pth")
    segresnet = models.load_pretrained_3D_UNET(path="./trained_models/SegResNet_best_metric_model.pth")
    unetr = models.load_pretrained_3D_UNET(path="./trained_models/UNETR_best_metric_model.pth")

    with torch.no_grad():
        img, mask = val_dataset[1]
        img, mask = img.to(device), mask.to(device)
        roi_size = (64, 64, 64)
        sw_batch_size = 4
        mask = (mask > 0).float()
        output_unetr = sliding_window_inference(
            img.unsqueeze(0), roi_size, sw_batch_size, unetr)
        output_unetr = [post_pred(i) for i in decollate_batch(output_unetr)]
        # mask_labels = [post_label(i) for i in decollate_batch(val_labels)]

        output_unet = sliding_window_inference(
            img.unsqueeze(0), roi_size, sw_batch_size, unet)
        output_unet = [post_pred(i) for i in decollate_batch(output_unet)]
        
        output_seg = sliding_window_inference(
            img.unsqueeze(0), roi_size, sw_batch_size, segnet)
        output_seg = [post_pred(i) for i in decollate_batch(output_seg)]
    
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(16,8))

    ax1.imshow(img[0][28,:,:].cpu(), cmap='gray')
    ax1.set_title('MRA Image')

    ax2.imshow(mask[0][28,:,:].cpu(), cmap='gray')
    ax2.set_title('Ground Truth')

    ax3.imshow(output_unet[0][1][28,:,:].cpu(), cmap='gray')
    ax3.set_title('3D UNet Mask')

    ax4.imshow(output_seg[0][1][28,:,:].cpu(), cmap='gray')
    ax4.set_title('SegResNet Mask')

    ax5.imshow(output_unetr[0][1][28,:,:].cpu(), cmap='gray')
    ax5.set_title('UNETR Mask')

    ax6.imshow(mask[0][28,:,:].cpu(), cmap='gray')
    ax6.set_title('Ground Truth')

    f.delaxes(ax6)
    
    f.savefig('output.png', bbox_inches=extent.expanded(1.1, 1.2))