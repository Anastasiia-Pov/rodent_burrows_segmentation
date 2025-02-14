from torch import nn
import torch
from segmentation_utils import make_conv_bn_relu

class UNetHalf(nn.Module):
    def __init__(self, in_channels, class_num):
        super().__init__()
        self.encoder1 = nn.Sequential(
            make_conv_bn_relu(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.encoder2 = nn.Sequential(
            make_conv_bn_relu(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.encoder3 = nn.Sequential(
            make_conv_bn_relu(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.encoder4 = nn.Sequential(
            make_conv_bn_relu(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.bottleneck = nn.Sequential(
            make_conv_bn_relu(in_channels=256, out_channels=512, kernel_size=3, padding=1),
        )

        self.upsample4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            make_conv_bn_relu(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            
        )
        self.decoder4 = nn.Sequential(
            make_conv_bn_relu(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            
        )

        self.upsample3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            make_conv_bn_relu(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            
        )
        self.decoder3 = nn.Sequential(
            make_conv_bn_relu(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            
        )

        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            make_conv_bn_relu(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            
        )
        self.decoder2 = nn.Sequential(
            make_conv_bn_relu(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            
        )

        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            make_conv_bn_relu(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            
        )
        self.decoder1 = nn.Sequential(
            make_conv_bn_relu(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            
        )

        self.out_conv = nn.Conv2d(in_channels=32, out_channels=class_num, kernel_size=3, padding=1)

        

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        h = self.bottleneck(self.pool4(enc4))

        dec4 = self.upsample4(h)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upsample3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upsample2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upsample1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        output = self.out_conv(dec1)
        
        return output

class UNetFull(nn.Module):
    def __init__(self, in_channels, class_num):
        super().__init__()
        self.encoder1 = nn.Sequential(
            make_conv_bn_relu(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=32, out_channels=32, kernel_size=3, padding=1),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.encoder2 = nn.Sequential(
            make_conv_bn_relu(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.encoder3 = nn.Sequential(
            make_conv_bn_relu(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.encoder4 = nn.Sequential(
            make_conv_bn_relu(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.bottleneck = nn.Sequential(
            make_conv_bn_relu(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        )

        self.upsample4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            make_conv_bn_relu(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        )
        self.decoder4 = nn.Sequential(
            make_conv_bn_relu(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        )

        self.upsample3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            make_conv_bn_relu(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        )
        self.decoder3 = nn.Sequential(
            make_conv_bn_relu(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        )

        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            make_conv_bn_relu(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        )
        self.decoder2 = nn.Sequential(
            make_conv_bn_relu(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        )

        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            make_conv_bn_relu(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=32, out_channels=32, kernel_size=3, padding=1),
        )
        self.decoder1 = nn.Sequential(
            make_conv_bn_relu(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        )

        self.out_conv = nn.Conv2d(in_channels=32, out_channels=class_num, kernel_size=3, padding=1)

        

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        h = self.bottleneck(self.pool4(enc4))

        dec4 = self.upsample4(h)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upsample3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upsample2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upsample1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        output = self.out_conv(dec1)
        
        return output

if __name__ == '__main__':
    model = UNetHalf(in_channels=3, class_num=2)
    print(model(torch.randn(1, 3, 256, 256)).shape)