from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn

from .Branch import MOD_Branch
from .DLA import MOD_DLA
from .ResNet import MOD_ResNet

backbone = {
    'dla': MOD_DLA,
    'resnet': MOD_ResNet
}

import torch
BN_MOMENTUM = 0.1

from .deconv import deconv_layers

import math

from .REFusion import REFusion


class RENet(nn.Module):
    def __init__(self, arch, num_layers1, num_layers2,):
        super(RENet, self).__init__()
        self.backbone1 = backbone[arch](num_layers1)
        self.backbone2 = backbone[arch](num_layers2)

        
        ## E-TMA
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2)
        self.maxpool_3 = nn.MaxPool2d(kernel_size=3)
        self.maxpool_4 = nn.MaxPool2d(kernel_size=4)
        self.coarse_fine_fusion = nn.Sequential(nn.Conv2d(64*3, 64, kernel_size=3, stride=1, padding=1, bias=True), nn.BatchNorm2d(64, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))

        self.conv1_2 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)


        self.middle_fusion_1_ful = nn.Sequential(nn.Conv2d(256+64, 256, kernel_size=3, stride=1, padding=1, bias=True), nn.BatchNorm2d(256, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))
        self.middle_fusion_2_ful = nn.Sequential(nn.Conv2d(512+256, 512, kernel_size=3, stride=1, padding=1, bias=True), nn.BatchNorm2d(512, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))
        self.middle_fusion_3_ful = nn.Sequential(nn.Conv2d(1024+512, 1024, kernel_size=3, stride=1, padding=1, bias=True), nn.BatchNorm2d(1024, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))
        self.middle_fusion_4_ful = nn.Sequential(nn.Conv2d(2048+1024, 2048, kernel_size=3, stride=1, padding=1, bias=True), nn.BatchNorm2d(2048, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))
        self.middle_fusion_5_ful = nn.Sequential(nn.Conv2d(64+64, 64, kernel_size=3, stride=1, padding=1, bias=True), nn.BatchNorm2d(64, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))
        self.deconv_layer_2 = deconv_layers(2048, BN_MOMENTUM)
        
        ## fuison
        self.fus = nn.ModuleList([
            REFusion(64, 64, 0),
            REFusion(256, 256, 1),
            REFusion(512, 512, 2),
            REFusion(1024, 1024, 3),
            REFusion(2048, 2048, 4),
            REFusion(64, 64, 5)
        ])

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    

    def forward(self, input):

        ## import pdb; pdb.set_trace()

        IMG = input[:, :3, :, :]
        EVT_15ms = input[:, 3:6, :, :]
        EVT_30ms = input[:, 6:9, :, :]
        EVT_50ms = input[:, 9:, :, :]

        chunk_image = self.conv1_1(IMG)
        chunk_image = self.backbone1.bn1(chunk_image)
        chunk_image = self.relu(chunk_image)
        chunk_image_0 = self.maxpool(chunk_image)
        chunk_image_1 = self.backbone1.layer1(chunk_image_0)
        chunk_image_2 = self.backbone1.layer2(chunk_image_1)
        chunk_image_3 = self.backbone1.layer3(chunk_image_2)
        chunk_image_4 = self.backbone1.layer4(chunk_image_3)
        chunk_image_5 = self.backbone1.deconv_layer.deconv_layers(chunk_image_4)


        ## E-TMA
        chunk_event_15ms = self.conv1_2(EVT_15ms)
        chunk_event_15ms = self.backbone2.bn1(chunk_event_15ms)
        chunk_event_15ms = self.relu(chunk_event_15ms)
        chunk_event_15ms = self.maxpool_2(chunk_event_15ms)

        chunk_event_30ms = self.conv1_2(EVT_30ms)
        chunk_event_30ms = self.backbone2.bn1(chunk_event_30ms)
        chunk_event_30ms = self.relu(chunk_event_30ms)
        chunk_event_30ms = self.maxpool_3(chunk_event_30ms)

        chunk_event_50ms = self.conv1_2(EVT_50ms)
        chunk_event_50ms = self.backbone2.bn1(chunk_event_50ms)
        chunk_event_50ms = self.relu(chunk_event_50ms)
        chunk_event_50ms = self.maxpool_4(chunk_event_50ms)


        chunk_event_30ms = nn.functional.interpolate(chunk_event_30ms, [72, 72])
        chunk_event_50ms = nn.functional.interpolate(chunk_event_50ms, [72, 72])

        chunk_event = torch.cat((chunk_event_15ms, chunk_event_30ms, chunk_event_50ms), dim=1)
        chunk_event_0 = self.coarse_fine_fusion(chunk_event)
        
        chunk_event_1 = self.backbone2.layer1(chunk_event_0)
        chunk_event_2 = self.backbone2.layer2(chunk_event_1)
        chunk_event_3 = self.backbone2.layer3(chunk_event_2)
        chunk_event_4 = self.backbone2.layer4(chunk_event_3)
        chunk_event_5 = self.backbone2.deconv_layer.deconv_layers(chunk_event_4)
        
        ## fuison
        chunk_0 = self.fus[0](chunk_image_0, chunk_event_0)
        
        chunk_1 = self.fus[1](chunk_image_1, chunk_event_1)
        chunk_1 = torch.cat((chunk_1, chunk_0), dim=1)
        chunk_1 = self.middle_fusion_1_ful(chunk_1)
        
        chunk_2 = self.fus[2](chunk_image_2, chunk_event_2)
        chunk_2 = torch.cat((chunk_2, self.maxpool(chunk_1)), dim=1)
        chunk_2 = self.middle_fusion_2_ful(chunk_2)
        
        chunk_3 = self.fus[3](chunk_image_3, chunk_event_3)
        chunk_3 = torch.cat((chunk_3, self.maxpool(chunk_2)), dim=1)
        chunk_3 = self.middle_fusion_3_ful(chunk_3)
        
        chunk_4 = self.fus[4](chunk_image_4, chunk_event_4)
        chunk_4 = torch.cat((chunk_4, self.maxpool(chunk_3)), dim=1)
        chunk_4 = self.middle_fusion_4_ful(chunk_4)
        
        chunk_5 = self.fus[5](chunk_image_5, chunk_event_5)
        chunk_5 = torch.cat((chunk_5, self.deconv_layer_2(chunk_4)), dim=1)
        chunk_5 = self.middle_fusion_5_ful(chunk_5)

        chunk = chunk_5
        
        return chunk


class MOD_Det(nn.Module):
    def __init__(self, backbone, branch_info, arch, head_conv, K):
        super(MOD_Det, self).__init__()

        self.K = K
        self.branch = MOD_Branch(backbone.backbone1.output_channel, arch, head_conv, branch_info, K)

    ## def forward(self, chunk1, chunk2):
    def forward(self, chunk1, chunk2=None):
        assert(self.K == len(chunk1))
        
        return [self.branch(chunk1)]
