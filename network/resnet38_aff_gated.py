import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F
import network.resnet38d
from tool import pyutils
import numpy as np


def _unfold(img, radius):
    assert img.dim() == 4, 'Unfolding requires NCHW batch'
    N, C, H, W = img.shape
    diameter = 2 * radius + 1
    return F.unfold(img, diameter, 1, radius).view(N, C, diameter, diameter, H, W)


class Net(network.resnet38d.Net):
    def __init__(self):
        super(Net, self).__init__()

        self.f8_3 = torch.nn.Conv2d(512, 64, 1, bias=False)
        self.f8_4 = torch.nn.Conv2d(1024, 128, 1, bias=False)
        self.f8_5 = torch.nn.Conv2d(4096, 256, 1, bias=False)

        self.f9 = torch.nn.Conv2d(448, 448, 1, bias=False)
        
        torch.nn.init.kaiming_normal_(self.f8_3.weight)
        torch.nn.init.kaiming_normal_(self.f8_4.weight)
        torch.nn.init.kaiming_normal_(self.f8_5.weight)
        torch.nn.init.xavier_uniform_(self.f9.weight, gain=4)

        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]

        self.from_scratch_layers = [self.f8_3, self.f8_4, self.f8_5, self.f9]

        self.predefined_featuresize = int(448//8)

        return

    def forward(self, x, radius = 4, to_dense=False):

        d = super().forward_as_dict(x)

        f8_3 = F.elu(self.f8_3(d['conv4']))
        f8_4 = F.elu(self.f8_4(d['conv5']))
        f8_5 = F.elu(self.f8_5(d['conv6']))
        x = F.elu(self.f9(torch.cat([f8_3, f8_4, f8_5], dim=1)))
        features = torch.cat([d['conv4'], d['conv5'], d['conv6']], dim=1)

        N, C, H, W = x.shape

        aff_crf = _unfold(x, radius=radius)
        aff_crf = torch.abs(aff_crf - aff_crf[:,:,radius,radius,:,:].view(N,C,1,1,H,W))
        aff_crf = torch.mean(aff_crf, dim=1)

        aff = torch.exp(-aff_crf)

        return features, aff, aff_crf


    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups



