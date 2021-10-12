import torch
import torchio as tio
from torch.utils.data import DataLoader, Dataset

from zipfile import ZipFile
from glob import glob
import os, re
import shutil
from tqdm import tqdm
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt

from datetime import datetime

from monai.data import ITKReader, PILReader, decollate_batch, CacheDataset
from  monai.transforms import *


import warnings
warnings.filterwarnings("ignore")

class MRADataset(Dataset):
    def __init__(self, df, transforms, mode='train'):
        super(MRADataset, self).__init__()
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.cropper = RandCropByLabelClasses(spatial_size=[64,64,64], num_classes=2, num_samples=1)
        self.scale_intensity = ScaleIntensityRange(a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        tof_img = self.df.iloc[idx]['tof_file']
        aneurysm_img = self.df.iloc[idx]['aneurysm_file']

        tof = sitk.ReadImage(tof_img)
        tof = sitk.GetArrayFromImage(tof)

        label = sitk.ReadImage(aneurysm_img)
        label = sitk.GetArrayFromImage(label)

        tof = self.transforms(tof)
        # tof = self.scale_intensity(tof)
        label = self.transforms(label)

        label = (label > 0).float()

        if self.mode == 'train':
            ntof = self.cropper(img=tof, label=label, image=tof)[0]
            nlabel = self.cropper(img=label, label=label, image=tof)[0]
            return ntof, nlabel
        else:
            return tof, label

        # label = self.post_label(label)

        # ntof = self.cropper(img=tof, label=label, image=tof)[0]
        # nlabel = self.cropper(img=label, label=label, image=tof)[0]


def get_datasets():
    train_transforms = Compose(transforms=[AddChannel(), ResizeWithPadOrCrop(spatial_size=[140, 512, 512], method='end'), Resize(spatial_size=[96, 96, 96]), ToTensor()])
    val_transforms = Compose(transforms=[AddChannel(), ResizeWithPadOrCrop(spatial_size=[140, 512, 512], method='end'), Resize(spatial_size=[96, 96, 96]), ToTensor()])

    train_dataset = MRADataset(df = pd.read_csv('./dataset/train_set.csv'), transforms = train_transforms, mode='train')
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

    val_dataset = MRADataset(df = pd.read_csv('./dataset/val_set.csv'), transforms = val_transforms, mode='val')
    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    return train_dataset, val_dataset

if __name__ == '__main__':
    print('This is an utility package which cannot be ran alone, import this another python script to use.')
