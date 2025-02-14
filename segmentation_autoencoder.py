from torch import nn
import torch
from segmentation_utils import make_conv_bn_relu


class SegmentationAutoencoderFull(nn.Module):
    def __init__(self, in_channels, class_num):
        super().__init__()
        self.encoder = nn.Sequential(
            make_conv_bn_relu(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            make_conv_bn_relu(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            make_conv_bn_relu(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            #make_conv_bn_relu(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            #make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        )

        self.decoder = nn.Sequential(
            #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            #make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            #make_conv_bn_relu(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            make_conv_bn_relu(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            make_conv_bn_relu(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=32, out_channels=class_num, kernel_size=3, padding=1)
        )

    def forward(self, x):
        h = self.encoder(x)
        out = self.decoder(h)
        return out

class SegmentationAutoencoderShallow(nn.Module):
    def __init__(self, in_channels, class_num):
        super().__init__()
        self.encoder = nn.Sequential(
            make_conv_bn_relu(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            make_conv_bn_relu(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            make_conv_bn_relu(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            #make_conv_bn_relu(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            #make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        )

        self.decoder = nn.Sequential(
            #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            #make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            #make_conv_bn_relu(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            make_conv_bn_relu(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            make_conv_bn_relu(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=32, out_channels=class_num, kernel_size=3, padding=1)
        )

    def forward(self, x):
        h = self.encoder(x)
        out = self.decoder(h)
        return out


class SegmentationAutoencoderShallowHalfReduced(nn.Module):
    def __init__(self, in_channels, class_num):
        super().__init__()
        self.encoder = nn.Sequential(
            make_conv_bn_relu(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            make_conv_bn_relu(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            make_conv_bn_relu(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            #make_conv_bn_relu(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            #make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        )

        self.decoder = nn.Sequential(
            #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            #make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            #make_conv_bn_relu(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            make_conv_bn_relu(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            make_conv_bn_relu(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=32, out_channels=class_num, kernel_size=3, padding=1)
        )

    def forward(self, x):
        h = self.encoder(x)
        out = self.decoder(h)
        return out
'''
class SegmentationAutoencoder(nn.Module):
    def __init__(self, in_channels, class_num):
        super().__init__()
        self.encoder = nn.Sequential(
            make_conv_bn_relu(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            make_conv_bn_relu(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            make_conv_bn_relu(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            make_conv_bn_relu(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2),
            make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2),
            make_conv_bn_relu(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2),
            make_conv_bn_relu(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2),
            make_conv_bn_relu(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=32, out_channels=class_num, kernel_size=3, padding=1)
        )

    def forward(self, x):
        h = self.encoder(x)
        out = self.decoder(h)
        return out


class SegmentationAutoencoder(nn.Module):
    def __init__(self, in_channels, class_num):
        super().__init__()
        self.encoder = nn.Sequential(
            make_conv_bn_relu(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            make_conv_bn_relu(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            make_conv_bn_relu(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            make_conv_bn_relu(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2),
            make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2),
            make_conv_bn_relu(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2),
            make_conv_bn_relu(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2),
            make_conv_bn_relu(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=32, out_channels=class_num, kernel_size=3, padding=1)
        )

    def forward(self, x):
        h = self.encoder(x)
        out = self.decoder(h)
        return out
'''
if __name__ == '__main__':
    cnn = SegmentationAutoencoder(3, 2)
    res = cnn(torch.randn(1, 3, 256, 256))
    print(res.shape)
