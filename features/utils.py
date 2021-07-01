import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F

from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import numpy as np
import random
from tqdm import tqdm
import os
from PIL import Image
from glob import glob
from sklearn.manifold import TSNE
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MVTec_dataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_path, target = self.img_paths[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.img_paths)


def gen_dataset(args):
    if args.dataset == 'capsule':
        base_dir = 'Data/capsule/'

        add_dir = 'train/good/'
        img_paths_train_good = glob(base_dir + add_dir + '*.png')
        labels_train_good = [0 for p in img_paths_train_good]

        add_dir = 'test/good/'
        img_paths_test_good = glob(base_dir + add_dir + '*.png')
        labels_test_good = [0 for p in img_paths_test_good]

        add_dir = 'test/crack/'
        img_paths_test_bad = glob(base_dir + add_dir + '*.png')
        add_dir = 'test/faulty_imprint/'
        img_paths_test_bad.extend(glob(base_dir + add_dir + '*.png'))
        add_dir = 'test/poke/'
        img_paths_test_bad.extend(glob(base_dir + add_dir + '*.png'))
        add_dir = 'test/scratch/'
        img_paths_test_bad.extend(glob(base_dir + add_dir + '*.png'))
        add_dir = 'test/squeeze/'
        img_paths_test_bad.extend(glob(base_dir + add_dir + '*.png'))
        labels_test_bad = [1 for p in img_paths_test_bad]

        ## Ground Truth
        add_dir = 'ground_truth/crack/'
        img_paths_ground_truth = glob(base_dir + add_dir + '*.png')
        add_dir = 'ground_truth/faulty_imprint/'
        img_paths_ground_truth.extend(glob(base_dir + add_dir + '*.png'))
        add_dir = 'ground_truth/scratch/'
        img_paths_ground_truth.extend(glob(base_dir + add_dir + '*.png'))
        add_dir = 'ground_truth/squeeze/'
        img_paths_ground_truth.extend(glob(base_dir + add_dir + '*.png'))
        add_dir = 'ground_truth/poke/'
        img_paths_ground_truth.extend(glob(base_dir + add_dir + '*.png'))

    if args.dataset == 'leather':
        base_dir = 'Data/leather/'

        add_dir = 'train/good/'
        img_paths_train_good = glob(base_dir + add_dir + '*.png')
        labels_train_good = [0 for p in img_paths_train_good]

        add_dir = 'test/good/'
        img_paths_test_good = glob(base_dir + add_dir + '*.png')
        labels_test_good = [0 for p in img_paths_test_good]

        add_dir = 'test/color/'
        img_paths_test_bad = glob(base_dir + add_dir + '*.png')
        add_dir = 'test/cut/'
        img_paths_test_bad.extend(glob(base_dir + add_dir + '*.png'))
        add_dir = 'test/fold/'
        img_paths_test_bad.extend(glob(base_dir + add_dir + '*.png'))
        add_dir = 'test/glue/'
        img_paths_test_bad.extend(glob(base_dir + add_dir + '*.png'))
        add_dir = 'test/poke/'
        img_paths_test_bad.extend(glob(base_dir + add_dir + '*.png'))
        labels_test_bad = [1 for p in img_paths_test_bad]

        ## Ground Truth
        add_dir = 'ground_truth/color/'
        img_paths_ground_truth = glob(base_dir + add_dir + '*.png')
        add_dir = 'ground_truth/cut/'
        img_paths_ground_truth.extend(glob(base_dir + add_dir + '*.png'))
        add_dir = 'ground_truth/fold/'
        img_paths_ground_truth.extend(glob(base_dir + add_dir + '*.png'))
        add_dir = 'ground_truth/glue/'
        img_paths_ground_truth.extend(glob(base_dir + add_dir + '*.png'))
        add_dir = 'ground_truth/poke/'
        img_paths_ground_truth.extend(glob(base_dir + add_dir + '*.png'))

    train_img_paths_train_good, test_img_paths_train_good, train_labels_train_good, test_labels_train_good = train_test_split(
        img_paths_train_good,
        labels_train_good,
        test_size=0.2, random_state=41, stratify=labels_train_good)

    train_img_paths_test_good, test_img_paths_test_good, train_labels_test_good, test_labels_test_good = train_test_split(
        img_paths_test_good,
        labels_test_good,
        test_size=0.2, random_state=41, stratify=labels_test_good)

    train_img_paths_test_bad, test_img_paths_test_bad, train_labels_test_bad, test_labels_test_bad = train_test_split(
        img_paths_test_bad,
        labels_test_bad,
        test_size=0.8, random_state=41, stratify=labels_test_bad)

    if args.learning == 'unsupervised':
        train_img_paths = train_img_paths_train_good + train_img_paths_test_good
        test_img_paths = test_img_paths_train_good + test_img_paths_test_good + test_img_paths_test_bad

        train_labels = train_labels_train_good + train_labels_test_good
        test_labels = test_labels_train_good + test_labels_test_good + test_labels_test_bad

    else:
        train_img_paths = train_img_paths_train_good + train_img_paths_test_good + train_img_paths_test_bad
        test_img_paths = test_img_paths_train_good + test_img_paths_test_good + test_img_paths_test_bad

        train_labels = train_labels_train_good + train_labels_test_good + train_labels_test_bad
        test_labels = test_labels_train_good + test_labels_test_good + test_labels_test_bad

    return train_img_paths,train_labels, test_img_paths, test_labels

mvtec_classes = []

def plot_tSNE(X, y):
    plt.figure(figsize=(8, 6))

    # clean the figure
    plt.clf()

    tsne = TSNE()
    X_embedded = tsne.fit_transform(X)
    cmap = ['#00FFFF', '#DEB887', '#B8860B', '#ffa58f', '#ff5f24']
    for idx in range(5):
        plt.scatter(X_embedded[(y == idx), 0], X_embedded[(y == idx), 1], c=cmap[idx], label=idx)

    for i in range(5):
        mvtec_classes.append(str(i))

    plt.legend(mvtec_classes, ncol=5, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(())
    plt.yticks(())

    plt.show()


def plot_tSNE_MVtec(model, metric_fc, train_loader):
    with torch.no_grad():
        model = model.cpu()
        metric_fc = metric_fc.cpu()
        _i = 0
        model.eval()
        for ii, data in enumerate(train_loader):
            _i = _i + 1
            # print(_i)
            data_input, label = data

            feature = model(data_input)
            ctgr = metric_fc(feature)
            ctgr = ctgr.argmax(dim=1)
            if _i == 1:
                plt_arc_x = feature
                plt_arc_y = label
                plt_arc_ctgr = ctgr
            else:
                plt_arc_x = torch.cat([plt_arc_x, feature], dim=0)
                plt_arc_y = torch.cat([plt_arc_y, label], dim=0)
                plt_arc_ctgr = torch.cat([plt_arc_ctgr, ctgr], dim=0)

        X = plt_arc_x.cpu().detach().numpy()
        y = plt_arc_y.cpu().detach().numpy()
        plot_tSNE(X, y)

        plt_arc_df = pd.DataFrame({'label': plt_arc_y})
        plt_arc_df['ctgr'] = plt_arc_ctgr

        X_train = X
        y_train = y
        return X, y, plt_arc_df