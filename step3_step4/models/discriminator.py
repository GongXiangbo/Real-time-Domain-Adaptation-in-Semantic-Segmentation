import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):

    def __init__(self, num_classes):
      super(Discriminator, self).__init__()
      self.layers = nn.Sequential(
          nn.Conv2d(in_channels=num_classes, out_channels=64, kernel_size=4, stride=2, padding=1),
          nn.LeakyReLU(negative_slope=0.2, inplace=True),
          nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
          nn.LeakyReLU(negative_slope=0.2, inplace=True),
          nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
          nn.LeakyReLU(negative_slope=0.2, inplace=True),
          nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
          nn.LeakyReLU(negative_slope=0.2, inplace=True),
          nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=2, padding=1),
          nn.Upsample(scale_factor=32, mode='bilinear')
      )

    def forward(self, x):
      x = self.layers(x)
      return x



class LightDiscriminator(nn.Module):

    def __init__(self, num_classes):
      super(LightDiscriminator, self).__init__()
      self.layers = nn.Sequential(
          DepthWiseSeparableConvolution(num_classes, 64),
          nn.LeakyReLU(negative_slope=0.2, inplace=True),
          DepthWiseSeparableConvolution(64, 128),
          nn.LeakyReLU(negative_slope=0.2, inplace=True),
          DepthWiseSeparableConvolution(128, 256),
          nn.LeakyReLU(negative_slope=0.2, inplace=True),
          DepthWiseSeparableConvolution(256, 512),
          nn.LeakyReLU(negative_slope=0.2, inplace=True),
          DepthWiseSeparableConvolution(512, 1),
          nn.Upsample(scale_factor=32, mode='bilinear')
      )

    def forward(self, x):
      x = self.layers(x)
      return x



class DepthWiseSeparableConvolution(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(DepthWiseSeparableConvolution, self).__init__()
        self.depth_wise = nn.Conv2d(ch_in, ch_in, kernel_size=4, stride=2, padding=1, groups=ch_in)
        self.point_wise = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        x = self.depth_wise(x)
        x = self.point_wise(x)
        return x
		
