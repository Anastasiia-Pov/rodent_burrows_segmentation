import torch
from torch import nn

def make_conv_bn_relu(in_channels, out_channels, kernel_size, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

def make_trans_conv_bn_relu(in_channels, out_channels, kernel_size, stride, dilation, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )