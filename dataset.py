import os
import csv
import glob
import cv2
import h5py
import torch
import random
import torch.utils.data
import torchio as tio
import pandas as pd
import xlrd
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
import warnings
warnings.filterwarnings('ignore')
from scipy.ndimage.interpolation import zoom
from sklearn.model_selection import train_test_split
from PIL import Image
from numpy import *
from scipy import ndimage
import numpy as np
import pdb
from sklearn import preprocessing

# Dataset class for HCC with clinical features
class hcc_3d_all(torch.utils.data.Dataset):
    def __init__(self, root, label_dir, usage="train", transform=None, valid_fold=0, sequ_name=["arterial"], around_s=15):
        super(hcc_3d_all, self).__init__()
        assert usage in ("train", "valid")
        self.root = root
        self.label_dir = label_dir
        self.usage = usage
        self.transform = transform
        self.valid_fold = valid_fold
        self.around_s = around_s
        self.sequ_name = sequ_name
        self.data = []
        self.label = []
        self.case = []
        self.name = []
        self.size = []
        self.core = []
        self.data_distribution = [0] * 3
        label_file_path = "case_label.csv"
        label_file = open(os.path.join(self.label_dir, label_file_path), "r")
        csv_reader = csv.reader(label_file)
        self.label_dict = {}
        for name, ZS, l, fold, _ in csv_reader:
            if name == "Name":
                continue
            if self.usage == "train" and int(fold) != self.valid_fold:
                name = str(ZS)
                self.label_dict[name] = int(l)
            elif self.usage == "valid" and int(fold) == self.valid_fold:
                name = str(ZS)
                self.label_dict[name] = int(l)
        print(len(self.label_dict))
        for l in self.sequ_name:
            self.get_data(l)
        print("load", usage, len(self.data))
        print(self.data_distribution)

    def get_target(self):
        return self.label

    def get_data(self, sequ_name):
        mr_list = sorted(glob.glob(os.path.join(self.root, sequ_name, "*" + sequ_name + ".npy")))
        print(os.path.join(self.root, sequ_name, "*" + sequ_name + ".npy"))
        count = 0
        mr_count = 0
        for idx, mr in enumerate(mr_list):
            name = mr.split("/")[-1].split("_")[:3]
            id = name[2]
            for l in name:
                if l[:2] == "ZS":
                    id = l[2:]
                    break
            if id not in self.label_dict:
                continue
            curmr = np.load(mr)
            mask = np.load(mr[:-4] + "_mask.npy", allow_pickle=True)[0]
            core_slice = mask.shape[2] / 2
            cy = mask.shape[1] / 2
            cx = mask.shape[0] / 2
            cube = curmr[0, int(cy) - int(self.height / 2):int(cy) + int(self.height / 2),
                   int(cx) - int(self.width / 2):int(cx) + int(self.width / 2),
                   int(core_slice) - int(self.layer_n / 2):int(core_slice) + int(self.layer_n / 2)]
            mask_cube = mask[int(cy) - int(self.height / 2):int(cy) + int(self.height / 2),
                        int(cx) - int(self.width / 2):int(cx) + int(self.width / 2),
                        int(core_slice) - int(self.layer_n / 2):int(core_slice) + int(self.layer_n / 2)]
            if cube.shape[0] >= self.width and cube.shape[1] >= self.height and cube.shape[2] >= self.layer_n:
                self.data.append(cube)
                self.label.append(self.label_dict[id])
                self.name.append(id)
                self.core.append([core_slice, cy, cx])
                struct = ndimage.generate_binary_structure(3, 2)
                around = (ndimage.binary_dilation(mask_cube, structure=struct, iterations=self.around_s) - mask_cube) * 1.5 + mask_cube
                self.mask.append(around)
                self.data_distribution[self.label_dict[id]] += 1
            else:
                print(cube.shape)

    def dilation(self, mask, dis):
        new_mask = np.zeros_like(mask)
        struct = ndimage.generate_binary_structure(2, 2)
        discrip = []
        for idx in range(mask.shape[0]):
            l = mask[idx, :, :]
            if l.max() > 0:
                discrip.append(idx)
            around = ndimage.binary_dilation(l, structure=struct, iterations=dis)
            new_mask[idx, :, :] = around
        zdis = np.ceil(dis / 3)
        down = min(discrip)
        up = max(discrip)
        for idx in range(new_mask.shape[0]):
            if idx < down:
                if down - idx <= zdis:
                    new_mask[idx, :, :] = new_mask[down, :, :]
            elif idx > up:
                if idx - up <= zdis:
                    new_mask[idx, :, :] = new_mask[up, :, :]
        return (new_mask - mask) * 1.5 + mask

    def getinfo(self):
        return self.name

    def __getitem__(self, index):
        patch = self.data[index].astype(np.float32)
        mask = self.mask[index].astype(np.float32)
        name = self.name[index]
        if self.transform is not None:
            patch = self.transform(patch)
        if self.usage == "train":
            if np.random.random() > 0.75:
                patch = cv2.flip(patch, 1)
                mask = cv2.flip(mask, 1)
            elif np.random.random() > 0.75:
                patch = cv2.flip(patch, 0)
                mask = cv2.flip(mask, 0)
            elif np.random.random() > 0.75:
                patch = cv2.flip(patch, -1)
                mask = cv2.flip(mask, -1)
            patch = np.array([patch])
            mask = np.array([mask])
            patch = self.Augmentation(patch)
            mask = self.Augmentation_form(mask)
            a = np.random.randint(0, patch.shape[1] - self.fheight + 1)
            b = np.random.randint(0, patch.shape[2] - self.fwidth + 1)
            c = np.random.randint(0, patch.shape[3] - self.flayer_n + 1)
            patch = patch[:, a:a + self.fheight, b:b + self.fwidth, c:c + self.flayer_n]
            mask = mask[:, a:a + self.fheight, b:b + self.fwidth, c:c + self.flayer_n]
        else:
            patch = np.array([patch])
            mask = np.array([mask])
            a = int((patch.shape[1] - self.fheight) / 2)
            b = int((patch.shape[2] - self.fwidth) / 2)
            c = int((patch.shape[3] - self.flayer_n) / 2)
            patch = patch[:, a:a + self.fheight, b:b + self.fwidth, c:c + self.flayer_n]
            mask = mask[:, a:a + self.fheight, b:b + self.fwidth, c:c + self.flayer_n]
        target = np.ones(3).astype(np.float32) * 0.033
        target[self.label[index]] = 0.933
        if self.return_mask:
            return torch.tensor(patch.astype(np.float32)), torch.tensor(target), torch.tensor(np.array(mask))
        else:
            return torch.tensor(patch.astype(np.float32)), torch.tensor(target)

    def __len__(self):
        return len(self.data)