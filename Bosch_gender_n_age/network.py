import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.models import resnet50
import math

class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, activ = nn.LeakyReLU(0.2), downsample = False, normalize = False, common = True):
        super(ResBlock, self).__init__()
        self.activ = activ
        self.downsample = downsample
        self.shortcut = dim_in != dim_out
        self.normalize = normalize
        
        self.conv_1 = nn.Conv2d(dim_in, dim_in, kernel_size = 3, stride = 1, padding = 1)
        self.conv_2 = nn.Conv2d(dim_in, dim_out, kernel_size = 3, stride = 1, padding = 1)
        if self.normalize:
            if common:
                self.norm_1 = nn.BatchNorm2d(dim_in)
                self.norm_2 = nn.BatchNorm2d(dim_in)
            else:
                self.norm_1 = nn.InstanceNorm2d(dim_in, affine = True)
                self.norm_2 = nn.InstanceNorm2d(dim_in, affine = True)

        if self.shortcut:
            self.conv_1x1 = nn.Conv2d(dim_in,dim_out, kernel_size = 1, stride = 1, bias=False)

    def _shortcut(self,x):
        if self.shortcut:
            x = self.conv_1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x,2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm_1(x)
        x = self.activ(x)
        x = self.conv_1(x)
        if self.downsample:
            x = F.avg_pool2d(x,2)
        if self.normalize:
            x = self.norm_2(x)
        x = self.activ(x)
        x = self.conv_2(x)
        return x
    
    def forward(self, x):
        out = self._residual(x)
        output = (out + self._shortcut(x))/math.sqrt(2)
        return output

class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()

        self.common_layers = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=7, stride = 2),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU(0.2),
                        nn.MaxPool2d(kernel_size=3, stride = 2),
                        ResBlock(64, 64, downsample=False, normalize=True),
                        ResBlock(64, 64, downsample=False, normalize=True),
                        ResBlock(64, 128, downsample=False, normalize=True),
                        ResBlock(128, 128, downsample=True, normalize=True),
                        ResBlock(128, 128, downsample=False, normalize=True),
                        ResBlock(128, 256, downsample=False, normalize=True),
                        ResBlock(256, 256, downsample=True, normalize=True),
                        ResBlock(256, 256, downsample=True, normalize=True),
                                )
        self.male = nn.Sequential(
                            ResBlock(256, 512, downsample=False, normalize=True, common = False),
                            ResBlock(512, 512, downsample=True, normalize=True, common = False),
                            ResBlock(512, 512, downsample=False, normalize=True, common = False),
                            nn.Conv2d(512, 1024, kernel_size = 2, stride = 2),
                            nn.Flatten(),
                            nn.Linear(1024, 1),
                            nn.ReLU())
        self.female = nn.Sequential(
                            ResBlock(256, 512, downsample=False, normalize=True, common = False),
                            ResBlock(512, 512, downsample=True, normalize=True, common = False),
                            ResBlock(512, 512, downsample=False, normalize=True, common = False),
                            nn.Conv2d(512, 1024, kernel_size = 2, stride = 2),
                            nn.Flatten(),
                            nn.Linear(1024, 1),
                            nn.ReLU())

    def forward(self, x, gender):
        x = self.common_layers(x)
        #print(self.female(x).shape)
        out = []
        out += [self.male(x)]
        out += [self.female(x)]
        out = torch.stack(out, dim = 1)
        #out = torch.gather(out, dim = 1, index= gender.unsqueeze(1))
        return out

class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        self.res = resnet50(pretrained=False)
        self.linear = nn.Sequential(
            nn.Linear(1000, 1),
            nn.ReLU()
        )

    def forward(self, x, g):
        out = self.res(x)
        out = self.linear(out)
        return out.unsqueeze(1)
    
