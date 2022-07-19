#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
# import torchvision
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.autoaugment import AutoAugment
from sklearn.model_selection import train_test_split


def split_dataset(df, label_col, seed=123, kfold=False):
    """
    :param df: dataframe with information of data
    :param seed: seed number for repeatability
    :param label_col:  The column with the labels
    :param kfold: boolean if True, it splits the dataste in
    training and test subsets. Training subset will be further split in K-Folds
    :return:
    """
    if not kfold:
        train, test = train_test_split(df, stratify=df[label_col], test_size=0.2,
                                       random_state=seed)
        val, test = train_test_split(test, stratify=test[label_col], test_size=0.5,
                                     random_state=seed)
        return train, val, test

    else:
        train, test = train_test_split(df, stratify=df[label_col], test_size=0.1, random_state=seed)
        
        return train, test


def get_transform_auto(train):
    """
    Resize image and apply augmentations default policy ImageNet
    """

    transforms = [T.Resize(size=(224, 224))]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        # transforms.append(TrivialAugmentWide())
        transforms.append(AutoAugment())
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return T.Compose(transforms)


def get_transform(train):
    """
    Resize image and apply augmentations
    """
    transforms = [T.Resize(size=(224, 224)), T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return T.Compose(transforms)


def get_vt_transform(train):
    """
    transformations for vision transformers
    """

    transforms = [T.Resize(size=(224, 224), interpolation=InterpolationMode.BICUBIC),
                  T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    # transforms = [T.Resize(size=(224, 224), interpolation=3),
    #               T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class CovidDataset(torch.utils.data.Dataset):
    """
    : process data to input into a dat loader
    :returns tensor images and labels and path to a data directory
    """

    def __init__(self, df, paths_col, labels_col, transforms=None):

        super(CovidDataset).__init__()
        self.df = df
        self.img_paths = paths_col
        self.labels = labels_col
        self.transforms = transforms
        self.dataset_files = [x for x in (self.df[self.img_paths].tolist()) if x.endswith('png')
                              or x.endswith('jpg')]
        self.classes = list(self.df[self.labels].unique())
        self.classes.sort()

        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}

        self.length = len(self.dataset_files)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        img_path = self.df.iloc[idx][self.img_paths]
        img = Image.open(img_path)
        label = self.df.iloc[idx][self.labels]
        lbl_idx = [v for k, v in self.class_to_idx.items() if k == label][0]
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)
        return img, lbl_idx, img_path