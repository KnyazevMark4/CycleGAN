import torch
from torch import nn
import numpy as np



def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class Discriminator(nn.Module):
    def __init__(self):
      super().__init__()
    
      # encoder (downsampling)
      self.enc_conv0 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
      )
      self.enc_conv1 = nn.Sequential(
          nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
          nn.InstanceNorm2d(128),
          nn.LeakyReLU(negative_slope=0.2, inplace=True),
      )
      self.enc_conv2 = nn.Sequential(
          nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
          nn.InstanceNorm2d(256),
          nn.LeakyReLU(negative_slope=0.2, inplace=True),
      )
      self.enc_conv3 = nn.Sequential(
          nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
          nn.InstanceNorm2d(512),
          nn.LeakyReLU(negative_slope=0.2, inplace=True),
      )
      self.enc_conv4 = nn.Sequential(
          nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
          nn.InstanceNorm2d(512),
          nn.Conv2d(512, 1, kernel_size=8, stride=1, padding=0),
          nn.Sigmoid(),
      )

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
      #################
      # print(input.shape)
      x = self.enc_conv0(input)
      x = self.enc_conv1(x)
      x = self.enc_conv2(x)
      x = self.enc_conv3(x)
      x = self.enc_conv4(x)
      return x
