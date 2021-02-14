import torch
from torch import nn
import numpy as np
from torchvision import transforms


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class Res_Block(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
      return self.module(inputs) + inputs

def R(num_in_features, num_out_features):
  return nn.Sequential(
            Res_Block(
                nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(num_in_features, num_out_features, kernel_size=3, stride=1, padding=0),
                    nn.InstanceNorm2d(num_out_features),
                    nn.ReLU(inplace=True),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(num_out_features, num_out_features, kernel_size=3, stride=1, padding=0),
                    nn.InstanceNorm2d(num_out_features),
                )
            ),
            nn.ReLU(inplace=True),
        )


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.pr0 = transforms.Normalize((0.485, 0.485, 0.485), (0.229, 0.229, 0.229)),

        # 256->256
        self.c1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # 256->128
        self.d2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # 128->64
        self.d3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # 64->64
        self.R4 = R(256, 256)
        # 64->64
        self.R5 = R(256, 256)
        # 64->64
        self.R6 = R(256, 256)
        # 64->64
        self.R7 = R(256, 256)
        # 64->64
        self.R8 = R(256, 256)
        # 64->64
        self.R9 = R(256, 256)
        # 64->64
        self.R10 = R(256, 256)
        # 64->64
        self.R11 = R(256, 256)
        # 64->64
        self.R12 = R(256, 256)
        # 64->129
        self.u13 = nn.Sequential(
            # nn.ReflectionPad2d(1),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=0, output_padding=0),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # 129->259
        self.u14 = nn.Sequential(
            # nn.ReflectionPad2d(1),
            nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=0, output_padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.c15 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(3),
            nn.Tanh()
        )
        

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):

        x1 = self.c1(x)
        x2 = self.d2(x1)

        x = self.d3(x2)
        x = self.R4(x)
        x = self.R5(x)
        x = self.R6(x)
        x = self.R7(x)
        x = self.R8(x)
        x = self.R9(x)
        x = self.R10(x)
        x = self.R11(x)
        x = self.R12(x)

        x = self.u13(x)
        x = torch.nn.Upsample(size=(128, 128), mode='bilinear')(x)
        x = self.u14(torch.cat([x,x2], 1))
        x = torch.nn.Upsample(size=(256, 256), mode='bilinear')(x)
        x = self.c15(torch.cat([x,x1], 1))

#         x = self.u13(x)
#         x = torch.nn.Upsample(size=(128, 128), mode='bilinear')(x)
#         x = self.u14(x)
#         x = torch.nn.Upsample(size=(256, 256), mode='bilinear')(x)
#         x = self.c15(x)

        x = (x+1)/2
                       
        return x
