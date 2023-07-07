"""
RNCapsGAN-VC model
Inspired by https://github.com/GANtastic3/MaskCycleGAN-VC
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from RNCapsGAN.layers import *

class Generator(nn.Module):
    """Generator of RNCapsGAN-VC
    """

    def __init__(self, input_shape=(80, 64), residual_in_channels=256):
        super(Generator, self).__init__()
        Cx, Tx = input_shape
        self.flattened_channels = (Cx // 4) * residual_in_channels
        #self.rn=RN_B(feature_channels=residual_in_channels)
        # 2D Conv Layer
        self.conv1 = nn.Conv2d(in_channels=2,
                               out_channels=residual_in_channels // 2,
                               kernel_size=(5, 15),
                               stride=(1, 1),
                               padding=(2, 7))

        self.conv1_gates = nn.Conv2d(in_channels=2,
                                     out_channels=residual_in_channels // 2,
                                     kernel_size=(5, 15),
                                     stride=1,
                                     padding=(2, 7))

        # 2D Downsampling Layers
        self.downSample1 = DownSampleGenerator(in_channels=residual_in_channels // 2,
                                               out_channels=residual_in_channels,
                                               kernel_size=5,
                                               stride=2,
                                               padding=2)

        self.downSample2 = DownSampleGenerator(in_channels=residual_in_channels,
                                               out_channels=residual_in_channels,
                                               kernel_size=5,
                                               stride=2,
                                               padding=2)

        # 2D -> 1D Conv
        self.conv2dto1dLayer = nn.Conv1d(in_channels=self.flattened_channels,
                                         out_channels=residual_in_channels,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)
        self.rn=RN_B(feature_channels=residual_in_channels)

        # Residual Blocks
        self.residualLayer1 = ResidualLayer(in_channels=residual_in_channels,
                                            out_channels=residual_in_channels * 2,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)
        self.residualLayer2 = ResidualLayer(in_channels=residual_in_channels,
                                            out_channels=residual_in_channels * 2,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)
        self.residualLayer3 = ResidualLayer(in_channels=residual_in_channels,
                                            out_channels=residual_in_channels * 2,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)
        self.residualLayer4 = ResidualLayer(in_channels=residual_in_channels,
                                            out_channels=residual_in_channels * 2,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)
        self.residualLayer5 = ResidualLayer(in_channels=residual_in_channels,
                                            out_channels=residual_in_channels * 2,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)
        self.residualLayer6 = ResidualLayer(in_channels=residual_in_channels,
                                            out_channels=residual_in_channels * 2,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)

        # 1D -> 2D Conv
        self.conv1dto2dLayer = nn.Conv1d(in_channels=residual_in_channels,
                                         out_channels=self.flattened_channels,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)
        self.rn_1 = RN_B(feature_channels=self.flattened_channels)

        # UpSampling Layers
        self.upSample1 = self.upsample(in_channels=residual_in_channels,
                                       out_channels=residual_in_channels * 4,
                                       kernel_size=5,
                                       stride=1,
                                       padding=2)

        self.glu = GLU()

        self.upSample2 = self.upsample(in_channels=residual_in_channels,
                                       out_channels=residual_in_channels * 2,
                                       kernel_size=5,
                                       stride=1,
                                       padding=2)

        # 2D Conv Layer
        self.lastConvLayer = nn.Conv2d(in_channels=residual_in_channels // 2,
                                       out_channels=1,
                                       kernel_size=(5, 15),
                                       stride=(1, 1),
                                       padding=(2, 7))

    def downsample(self, in_channels, out_channels, kernel_size, stride, padding):
        self.ConvLayer = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                                       nn.InstanceNorm1d(
                                           num_features=out_channels,
                                           affine=True),
                                       GLU())

        return self.ConvLayer

    def upsample(self, in_channels, out_channels, kernel_size, stride, padding):
        self.convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                                       nn.PixelShuffle(upscale_factor=2),
                                       nn.InstanceNorm2d(
                                           num_features=out_channels // 4,
                                           affine=True),
                                       GLU())
        return self.convLayer

    def forward(self, x, mask):
        # Conv2d
        x = torch.stack((x*mask, mask), dim=1)
        #x=x*mask
        conv1 = self.conv1(x) * torch.sigmoid(self.conv1_gates(x))  # GLU

        # Downsampling
        downsample1 = self.downSample1(conv1)
        downsample2 = self.downSample2(downsample1)

        # Reshape
        reshape2dto1d = downsample2.view(
            downsample2.size(0), self.flattened_channels, 1, -1)
        reshape2dto1d = reshape2dto1d.squeeze(2)

        # 2D -> 1D
        conv2dto1d_layer = self.conv2dto1dLayer(reshape2dto1d)
        conv2dto1d_layer = self.rn(conv2dto1d_layer,mask)

        # Residual Blocks
        residual_layer_1 = self.residualLayer1(conv2dto1d_layer)
        residual_layer_2 = self.residualLayer2(residual_layer_1)
        residual_layer_3 = self.residualLayer3(residual_layer_2)
        residual_layer_4 = self.residualLayer4(residual_layer_3)
        residual_layer_5 = self.residualLayer5(residual_layer_4)
        residual_layer_6 = self.residualLayer6(residual_layer_5)

        # 1D -> 2D
        conv1dto2d_layer = self.conv1dto2dLayer(residual_layer_6)
        conv1dto2d_layer = self.rn_1(conv1dto2d_layer,mask)

        # Reshape
        reshape1dto2d = conv1dto2d_layer.unsqueeze(2)
        reshape1dto2d = reshape1dto2d.view(reshape1dto2d.size(0), 256, 20, -1)

        # UpSampling
        upsample_layer_1 = self.upSample1(reshape1dto2d)
        upsample_layer_2 = self.upSample2(upsample_layer_1)

        # Conv2d
        output = self.lastConvLayer(upsample_layer_2)
        output = output.squeeze(1)
        return output


class Discriminator(nn.Module):
    """PatchGAN discriminator.
    """

    def __init__(self, input_shape=(80, 64), residual_in_channels=32):
        super(Discriminator, self).__init__()

        self.convLayer1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                                  out_channels=residual_in_channels // 2,
                                                  kernel_size=(3, 3),
                                                  stride=(1, 1),
                                                  padding=(1, 1)),
                                        GLU())

        # Downsampling Layers
        self.downSample1 = self.downsample(in_channels=residual_in_channels // 2,
                                           out_channels=residual_in_channels,
                                           kernel_size=(3, 3),
                                           stride=(2, 2),
                                           padding=1)

        self.downSample2 = self.downsample(in_channels=residual_in_channels,
                                           out_channels=residual_in_channels * 2,
                                           kernel_size=(3, 3),
                                           stride=[2, 2],
                                           padding=1)

        self.downSample3 = self.downsample(in_channels=residual_in_channels * 2,
                                           out_channels=residual_in_channels * 4,
                                           kernel_size=[3, 3],
                                           stride=[2, 2],
                                           padding=1)

        # self.downSample4 = self.downsample(in_channels=residual_in_channels * 4,
        #                                    out_channels=residual_in_channels * 4,
        #                                    kernel_size=[1, 10],
        #                                    stride=(1, 1),
        #                                    padding=(0, 2))

        # Conv Layer
        self.outputConvLayer = nn.Sequential(nn.Conv2d(in_channels=residual_in_channels * 2,
                                                       out_channels=64,
                                                       kernel_size=(1, 3),
                                                       stride=[2, 2],
                                                       padding=[0, 1]))

        self.conv_layer = ConvLayer()
        self.primary_capsules = PrimaryCaps()
        self.fc_capsules = FCCaps()

    def downsample(self, in_channels, out_channels, kernel_size, stride, padding):
        convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding),
                                  nn.InstanceNorm2d(num_features=out_channels,
                                                    affine=True),
                                  GLU())
        return convLayer

    def forward(self, x):
        # x has shape [batch_size, num_features, frames]
        # discriminator requires shape [batchSize, 1, num_features, frames]
        x = x.unsqueeze(1)
        conv_layer_1 = self.convLayer1(x)
        print(conv_layer_1.shape)
        downsample1 = self.downSample1(conv_layer_1)
        print(downsample1.shape)
        downsample2 = self.downSample2(downsample1)
        #downsample3 = self.downSample3(downsample2)
        print(downsample2.shape )
        fv=self.outputConvLayer(downsample2)
        print(fv.shape)
        primary_caps_output = self.primary_capsules(fv)
        print(primary_caps_output.shape)
        # # pco = primary_caps_output.unsqueeze(2)
        # # print(pco.shape)
        caps_output = self.fc_capsules(primary_caps_output).squeeze(0).transpose(0, 1)
        print(caps_output.shape)


        output = torch.flatten(caps_output)
        print(output.shape)
        return output

if __name__ == '__main__':
    # Non exhaustive test for RNCapsGAN-VC models

    # Generator Dimensionality Testing
    np.random.seed(0)

    residual_in_channels = 256
    input = np.random.randn(2, 80, 64)
    input = np.random.randn(2, 80, 64)
    input = torch.from_numpy(input).float()
    print("Generator input: ", input.shape)
    mask = torch.ones_like(input)
    # #mask.to("cuda:0")
    generator = Generator(input.shape[1:], residual_in_channels)
    output = generator(input, mask)
    print("Generator output shape: ", output.shape)

    #Discriminator Dimensionality Testing
    discriminator = Discriminator(output.shape[1:], residual_in_channels=32)
    output = discriminator(output)
    #print("Discriminator output shape ", output)
