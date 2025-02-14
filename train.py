import torch
from torch import nn
import torchvision

from sklearn.model_selection import train_test_split

from torchvision.models.segmentation import fcn

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn import metrics

import os

import pandas as pd

import glob

import random
import os
import numpy as np

import argparse

from datasets import SegmentationDataset
from trainer import SegmentationTrainer

from segmentation_autoencoder import *
from unet import *
from segnet import *

import segmentation_models_pytorch as smp


import warnings
warnings.filterwarnings("ignore")


class SegmentationWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)['out']


class ConfusionMatrix:
    def __init__(self, labels):
        self.labels = labels

    def __call__(self, true, pred):
        return metrics.confusion_matrix(true, pred, labels=self.labels)

class ConfusionIoU:
    def __init__(self, labels):
        self.labels = labels
    def __call__(self, confusion_matrix, **kwargs):
        iou = np.zeros(shape=(len(self.labels),))
        for label in self.labels:
            tp =confusion_matrix[label, label]
            #tn = np.sum(np.diag(confusion_matrix)) - tp
            fp = np.sum(confusion_matrix[label]) - tp
            fn = np.sum(confusion_matrix[:, label]) - tp

            if tp + fp + fn == 0:
                iou[label] = 0
            else:
                iou[label] = tp / (tp + fp + fn)
        return iou

class MeanConfuaionIou(ConfusionIoU):
    def __call__(self, confusion_matrix, **kwargs):
        return np.mean(super().__call__(confusion_matrix, **kwargs))


class ConfusionAccuracy:
    def __call__(self, confusion_matrix, **kwargs):
        return np.sum(np.diag(confusion_matrix))/np.sum(confusion_matrix)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_dataset',  required=True, help='Path to training datset')
    parser.add_argument('--resume_training', action='store_true', help='Restore training from checkpoint')
    parser.add_argument('--path_to_checkpoint', help='Path to saved checkpoint of the model')
    parser.add_argument('--batch_size', required=True, type=int, help='Batch size during trainig of the neural network')
    parser.add_argument('--nn_input_size', nargs='+', type=int, help='Sizes of the input images, columns_num, rows_num')
    parser.add_argument('--epoch_num', type=int, help='A number of thrainig epochs')
    parser.add_argument('--test_size_portion', type=float, help='')
    

    
    sample_args = [
        '--path_to_dataset',
        '../mice_holes_dataset',
        '--batch_size', '16',
        '--epoch_num', '2000',
        '--nn_input_size', '256', '256',
        '--test_size_portion', '0.5']

    args = parser.parse_args(sample_args)

    path_to_dataset = args.path_to_dataset
    nn_input_size = args.nn_input_size
    resume_training = args.resume_training
    path_to_checkpoint = args.path_to_checkpoint
    epoch_num = int(args.epoch_num)
    test_size_portion = args.test_size_portion

    class_num = 2
    
    if resume_training == True:
        if path_to_checkpoint is None:
            raise ValueError('--path_to_checkpoint flag must be specified if --resume_trining flag')
        

    names_list = sorted([f[:-4] for f in os.listdir(os.path.join(path_to_dataset, 'images')) if f.endswith('.jpg')])#[:50]

    train_names, test_names = train_test_split(names_list, test_size=test_size_portion, random_state=0)

    train_transforms = A.Compose(
        [
            A.Resize(*nn_input_size),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Affine(scale=(0.7, 1.3), rotate=(-90,90), shear=(-15, 15)),
            A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
            ToTensorV2()
        ]
    )

    test_transforms = A.Compose(
        [
            A.Resize(*nn_input_size),
            A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
            ToTensorV2()
        ]
    )

    batch_size=16

    train_dataset = SegmentationDataset(path_to_dataset=path_to_dataset, instance_names_list=train_names, transforms=train_transforms)
    test_dataset = SegmentationDataset(path_to_dataset=path_to_dataset, instance_names_list=test_names, transforms=test_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model = SegmentationAutoencoderShallowHalfReduced(in_channels=3, class_num=2)
    model_name = 'seg_ae_shallow_half_2000ep'

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters())

    print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')

    # вычислим веса классов
    path_to_csv = os.path.join(path_to_dataset, 'overall_pixel_stats.csv')
    stat_df = pd.read_csv(path_to_csv).drop(['Название изображения'], axis=1)
    
    weights = (stat_df.sum(axis=0)).to_numpy()
    #weights = 1-weights/weights.sum()
    weights = weights/weights.sum()
    weights = torch.FloatTensor(weights).to(device)
        
    criterion = nn.CrossEntropyLoss(weight=weights)
    #criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    #smp.losses.FocalLoss(mode='multiclass')

    labels = np.arange(0, class_num)
    metrics_dict = {
        'loss': None,
        'confusion_matrix': ConfusionMatrix(labels=labels),
        'accuracy': ConfusionAccuracy(),
        'class IoU': ConfusionIoU(labels=labels),
        'mean IoU': MeanConfuaionIou(labels=labels),
    }

    metrics_to_dispaly = ['loss', 'accuracy', 'mean IoU', 'class IoU']

    segmentation_trainer = SegmentationTrainer(
        model=model,
        model_name=model_name,
        train_loader=train_loader,
        test_loader=test_loader,
        metrics_dict=metrics_dict,
        metrics_to_display=metrics_to_dispaly,
        criterion=criterion,
        optimizers_list=[optimizer],
        checkpoint_criterion='mean IoU',
        device=device)

    if resume_training:
        segmentation_trainer = torch.load(path_to_checkpoint)
        segmentation_trainer.train_loader.dataset.path_to_dataset = path_to_dataset
        segmentation_trainer.train_loader.dataset.path_to_dataset = path_to_dataset

    segmentation_trainer.train(epoch_num)
    epoch_idx = segmentation_trainer.testing_log_df['mean IoU'].astype(float).argmax()

    best_mean_iou = segmentation_trainer.testing_log_df['mean IoU'].astype(float).max()
    best_class_iou = segmentation_trainer.testing_log_df['class IoU']

    print('Best IoU for {} is {} on {} epoch'.format(model_name, best_mean_iou, epoch_idx))