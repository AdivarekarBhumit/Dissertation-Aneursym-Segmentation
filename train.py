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
    
    # Load the datasets
    train_dataset, val_dataset = get_datasets()

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Load models
    if config.MODEL_TO_TRAIN == 'UNET':
        model = models.get_3D_UNET()
    elif config.MODEL_TO_TRAIN == 'SegResNet':
        model = models.get_SegResNet()
    elif config.MODEL_TO_TRAIN == 'UNETR':
        model = models.get_UNETR()

    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    dice_metric = DiceMetric(include_background=True, reduction="mean")

    # Training and Validation code
    root_dir = "./best_models/"
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    max_epochs = config.NUM_EPOCHS
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    train_dice = []
    metric_values = []
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=True, num_classes=2)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=True, num_classes=2)])

    roi_size = (64, 64, 64)
    sw_batch_size = 1

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            # inputs, labels = (
            #     batch_data[0]["image"].to(device),
            #     batch_data[0]["label"].to(device),
            # )
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            labels = (labels > 0).float()
            # print('Shapes:', inputs.shape, outputs.shape, labels.shape)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # Training dice score
            with torch.no_grad():
                train_outputs = sliding_window_inference(
                    inputs, roi_size, sw_batch_size, model)
                train_outputs = [post_pred(i) for i in decollate_batch(train_outputs)]
                train_labels = [post_label(i) for i in decollate_batch(labels)]
                # compute metric for current iteration
                dice_metric(y_pred=train_outputs, y=train_labels)
        
        # aggregate the final mean dice result
        metric = dice_metric.aggregate().item()
        # reset the status for next validation round
        dice_metric.reset()
        train_dice.append(metric)
   
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1}, average loss: {epoch_loss:.4f}, average training dice:{metric:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = val_data[0].to(device), val_data[1].to(device)
                    roi_size = (64, 64, 64)
                    sw_batch_size = 2
                    val_labels = (val_labels > 0).float()
                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size, sw_batch_size, model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        root_dir, f"{config.MODEL_TO_TRAIN}_best_metric_model.pth"))
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )
