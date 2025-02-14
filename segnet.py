from torch import nn
import torch
from segmentation_utils import make_conv_bn_relu

import torchvision


class SegNetFull(nn.Module):
    def __init__(self, in_channels, class_num):
        super().__init__()
        self.encoder1 = nn.Sequential(
            make_conv_bn_relu(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=32, out_channels=32, kernel_size=3, padding=1),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        self.encoder2 = nn.Sequential(
            make_conv_bn_relu(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        self.encoder3 = nn.Sequential(
            make_conv_bn_relu(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        self.encoder4 = nn.Sequential(
            make_conv_bn_relu(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        self.bottleneck = nn.Sequential(
            make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        )

        self.unpool4 = nn.MaxUnpool2d(kernel_size=2)
        self.decoder4 = nn.Sequential(
            make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        )
        

        self.unpool3 = nn.MaxUnpool2d(kernel_size=2)
        self.decoder3 = nn.Sequential(
            make_conv_bn_relu(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        )
        

        self.unpool2 = nn.MaxUnpool2d(kernel_size=2)
        self.decoder2 = nn.Sequential(
            make_conv_bn_relu(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        )
        

        self.unpool1 = nn.MaxUnpool2d(kernel_size=2)
        self.decoder1 = nn.Sequential(
            make_conv_bn_relu(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            make_conv_bn_relu(in_channels=32, out_channels=32, kernel_size=3, padding=1),
        )

        self.out_block = nn.Sequential(
            make_conv_bn_relu(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=32, out_channels=class_num, kernel_size=3, padding=1)
        )

        

    def forward(self, x):
        enc, indices1 = self.pool1(self.encoder1(x))
        #enc1_size = enc.size()

        enc, indices2 = self.pool2(self.encoder2(enc))
        #enc2_size = enc.size()

        enc, indices3 = self.pool3(self.encoder3(enc))
        #enc = self.encoder3(enc)
        #enc3_size = enc.size()
        

        enc, indices4 = self.pool4(self.encoder4(enc))
        #enc4_size = enc.size()
        #return enc

        enc = self.bottleneck(enc)
        #return enc


        dec = self.decoder4(enc)
        #return dec
        
        
        dec = self.unpool4(dec, indices4)
        #return dec
        dec = self.decoder3(dec)
        
        

        dec = self.unpool3(dec, indices3)
        #return dec
        dec = self.decoder2(dec)
        
        

        dec = self.unpool2(dec, indices2)
        dec = self.decoder1(dec)

        dec = self.unpool1(dec, indices1)
        dec = self.out_block(dec)
        
        return dec


class SegNetHalf(nn.Module):
    def __init__(self, in_channels, class_num):
        super().__init__()
        self.encoder1 = nn.Sequential(
            make_conv_bn_relu(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=32, out_channels=32, kernel_size=3, padding=1),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        self.encoder2 = nn.Sequential(
            make_conv_bn_relu(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        self.encoder3 = nn.Sequential(
            make_conv_bn_relu(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        self.encoder4 = nn.Sequential(
            make_conv_bn_relu(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        self.bottleneck = nn.Sequential(
            make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        )

        self.unpool4 = nn.MaxUnpool2d(kernel_size=2)
        self.decoder4 = nn.Sequential(
            make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        )
        

        self.unpool3 = nn.MaxUnpool2d(kernel_size=2)
        self.decoder3 = nn.Sequential(
            make_conv_bn_relu(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        )
        

        self.unpool2 = nn.MaxUnpool2d(kernel_size=2)
        self.decoder2 = nn.Sequential(
            make_conv_bn_relu(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        )
        

        self.unpool1 = nn.MaxUnpool2d(kernel_size=2)
        self.decoder1 = nn.Sequential(
            make_conv_bn_relu(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            #make_conv_bn_relu(in_channels=32, out_channels=32, kernel_size=3, padding=1),
        )

        self.out_block = nn.Sequential(
            #make_conv_bn_relu(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=32, out_channels=class_num, kernel_size=3, padding=1)
        )

        

    def forward(self, x):
        enc, indices1 = self.pool1(self.encoder1(x))
        #enc1_size = enc.size()

        enc, indices2 = self.pool2(self.encoder2(enc))
        #enc2_size = enc.size()

        enc, indices3 = self.pool3(self.encoder3(enc))
        #enc = self.encoder3(enc)
        #enc3_size = enc.size()
        

        enc, indices4 = self.pool4(self.encoder4(enc))
        #enc4_size = enc.size()
        #return enc

        enc = self.bottleneck(enc)
        #return enc


        dec = self.decoder4(enc)
        #return dec
        
        
        dec = self.unpool4(dec, indices4)
        #return dec
        dec = self.decoder3(dec)
        
        

        dec = self.unpool3(dec, indices3)
        #return dec
        dec = self.decoder2(dec)
        
        

        dec = self.unpool2(dec, indices2)
        dec = self.decoder1(dec)

        dec = self.unpool1(dec, indices1)
        dec = self.out_block(dec)
        
        return dec



class VGGBNEncoder(nn.Module):
    def __init__(self, pretrained_model, blocks_num=5) -> None:
        super().__init__()

        self.blocks_list = nn.ModuleList()

        self.encoder_channels_num_list = []

        block = []
        actual_blocks_num = 0
        for layer in pretrained_model:
            if type(layer)!=torch.nn.modules.pooling.MaxPool2d:
                block.append(layer)
                if type(layer)==nn.Conv2d:
                    out_channels_num = layer.out_channels
                #print('AAAAA')
            else:
                block.append(nn.MaxPool2d(kernel_size=2, return_indices=True))

                self.blocks_list.append(nn.Sequential(*block))
                self.encoder_channels_num_list.append(out_channels_num)
                block = []
                actual_blocks_num += 1
                if actual_blocks_num == blocks_num:
                    break
        self.encoder_channels_num_list = self.encoder_channels_num_list[::-1]
        
    def forward(self, x):
        indices_list = []
        for block in self.blocks_list:
            x, indices = block(x)
            indices_list.append(indices)

        return x, indices_list[::-1]


class PretrainedSegNet(nn.Module):
    def __init__(self, pretrained_model, class_num, encoder_blocks_num=5):
        super().__init__()

        self.encoder = VGGBNEncoder(pretrained_model, blocks_num=encoder_blocks_num)
        self.encoder_channels_num_list = self.encoder.encoder_channels_num_list
        '''
        self.output_classifier = nn.Sequential(
            make_conv_bn_relu(in_channels=self.encoder_channels_num_list[-1], out_channels=32, kernel_size=3, padding=1)
            nn.Conv2d(in_channels=32, out_channels=class_num, kernel_size=3, padding=1)
        )
        '''
        #self.output_classifier = nn.Conv2d(in_channels=64, out_channels=class_num, kernel_size=3, padding=1)

        self.decoder_blocks_list = nn.ModuleList()

        for idx in range(len(self.encoder_channels_num_list)-1):
            in_channels_num = self.encoder_channels_num_list[idx]
            out_channels_num = self.encoder_channels_num_list[idx+1]

            #self.decoder_blocks_list.append(MaxUnpoolWrapper(kernel_size=2))
            self.decoder_blocks_list.append(nn.Sequential(
                make_conv_bn_relu(in_channels=in_channels_num, out_channels=out_channels_num, kernel_size=3, padding=1),
                make_conv_bn_relu(in_channels=out_channels_num, out_channels=out_channels_num, kernel_size=3, padding=1)
            ))

        self.decoder_blocks_list.append(nn.Sequential(
                make_conv_bn_relu(in_channels=out_channels_num, out_channels=32, kernel_size=3, padding=1),
                make_conv_bn_relu(in_channels=32, out_channels=class_num, kernel_size=3, padding=1)
            ))

    def forward(self, x):
        h, indices_list = self.encoder(x)
        for decoder, indices in zip(self.decoder_blocks_list, indices_list):
            h = nn.functional.max_unpool2d(h, indices, kernel_size=2)
            h = decoder(h)

        return h



if __name__ == '__main__':
    model = SegNetHalf(in_channels=3, class_num=2)
    print(model(torch.randn(1, 3, 256, 256)).shape)