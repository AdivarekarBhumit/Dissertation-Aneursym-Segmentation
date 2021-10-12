import torch
import torchio as tio
from torch.utils.data import DataLoader, Dataset

from zipfile import ZipFile
from glob import glob
import os, re
import shutil
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt

from datetime import datetime

from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR, UNet, DynUNet, SegResNet
from monai.inferers import sliding_window_inference
from monai.networks.layers import Norm

import warnings
warnings.filterwarnings("ignore")

def get_3D_UNET():
    device = torch.device("cuda:0")
    model = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

    return model

def get_SegResNet():
    device = torch.device("cuda:0")
    
    # model = DynUNet(
    #     spatial_dims = 3,
    #     in_channels = 1,
    #     out_channels = 2,
    #     norm_name = Norm.BATCH,
    #     kernel_size = (5,5,5),
    #     strides = (1,2,2),
    #     upsample_kernel_size = (2,2,2),
    #     res_block=True
    # ).to(device)

    model = SegResNet(act='relu', upsample_mode='deconv').to(device)

    return model

def get_UNETR():
    device = torch.device("cuda:0")
    model = UNETR(
        in_channels=1,
        out_channels=2,
        img_size=(64,64,64),
        feature_size=16,
        hidden_size=256,
        mlp_dim=1024,
        num_heads=8,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.5,
    ).to(device)

    return model

def load_pretrained_3D_UNET(path=None):
    if path == None:
        raise "Please provide path to the saved model"
    else:
        model = get_3D_UNET()
        model.load_state_dict(torch.load(path))
        return model

def load_pretrained_SegResNet(path=None):
    if path == None:
        raise "Please provide path to the saved model"
    else:
        model = get_SegResNet()
        model.load_state_dict(torch.load(path))
        return model
    
def load_pretrained_UNETR(path=None):
    if path == None:
        raise "Please provide path to the saved model"
    else:
        model = get_UNETR()
        model.load_state_dict(torch.load(path))
        return model

