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


class MOD_Net(nn.Module):
    def __init__(self, arch, num_layers1, num_layers2, branch_info, head_conv, K):
        super(MOD_Net, self).__init__()
        
        self.K = K

        self.backbone1 = backbone[arch](num_layers1)
        self.backbone2 = backbone[arch](num_layers2)
        
        self.branch = MOD_Branch(self.backbone1.output_channel, arch, head_conv, branch_info, K)

        
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
        IMG = [input[i][:, :3, :, :] for i in range(self.K)]
        EVT_15ms = [input[i][:, 3:6, :, :] for i in range(self.K)]
        EVT_30ms = [input[i][:, 6:9, :, :] for i in range(self.K)]
        EVT_50ms = [input[i][:, 9:, :, :] for i in range(self.K)]
        
        chunk_image = [self.conv1_1(IMG[i]) for i in range(self.K)]
        chunk_image = [self.backbone1.bn1(chunk_image[i]) for i in range(self.K)]
        chunk_image = [self.relu(chunk_image[i]) for i in range(self.K)]
        chunk_image_0 = [self.maxpool(chunk_image[i]) for i in range(self.K)]
        chunk_image_1 = [self.backbone1.layer1(chunk_image_0[i]) for i in range(self.K)]
        chunk_image_2 = [self.backbone1.layer2(chunk_image_1[i]) for i in range(self.K)]
        chunk_image_3 = [self.backbone1.layer3(chunk_image_2[i]) for i in range(self.K)]
        chunk_image_4 = [self.backbone1.layer4(chunk_image_3[i]) for i in range(self.K)]
        chunk_image_5 = [self.backbone1.deconv_layer.deconv_layers(chunk_image_4[i]) for i in range(self.K)]


        ## E-TMA
        chunk_event_15ms = [self.conv1_2(EVT_15ms[i]) for i in range(self.K)]
        chunk_event_15ms = [self.backbone2.bn1(chunk_event_15ms[i]) for i in range(self.K)]
        chunk_event_15ms = [self.relu(chunk_event_15ms[i]) for i in range(self.K)]
        chunk_event_15ms = [self.maxpool_2(chunk_event_15ms[i]) for i in range(self.K)]
        
        chunk_event_30ms = [self.conv1_2(EVT_30ms[i]) for i in range(self.K)]
        chunk_event_30ms = [self.backbone2.bn1(chunk_event_30ms[i]) for i in range(self.K)]
        chunk_event_30ms = [self.relu(chunk_event_30ms[i]) for i in range(self.K)]
        chunk_event_30ms = [self.maxpool_3(chunk_event_30ms[i]) for i in range(self.K)]

        chunk_event_50ms = [self.conv1_2(EVT_50ms[i]) for i in range(self.K)]
        chunk_event_50ms = [self.backbone2.bn1(chunk_event_50ms[i]) for i in range(self.K)]
        chunk_event_50ms = [self.relu(chunk_event_50ms[i]) for i in range(self.K)]
        chunk_event_50ms = [self.maxpool_4(chunk_event_50ms[i]) for i in range(self.K)]
        
        chunk_event_30ms = [nn.functional.interpolate(chunk_event_30ms[i], [72, 72]) for i in range(self.K)]
        chunk_event_50ms = [nn.functional.interpolate(chunk_event_50ms[i], [72, 72]) for i in range(self.K)]

        chunk_event = [torch.cat((chunk_event_15ms[i], chunk_event_30ms[i], chunk_event_50ms[i]), dim=1) for i in range(self.K)]
        chunk_event_0 = [self.coarse_fine_fusion(chunk_event[i]) for i in range(self.K)]

        chunk_event_1 = [self.backbone2.layer1(chunk_event_0[i]) for i in range(self.K)]
        chunk_event_2 = [self.backbone2.layer2(chunk_event_1[i]) for i in range(self.K)]
        chunk_event_3 = [self.backbone2.layer3(chunk_event_2[i]) for i in range(self.K)]
        chunk_event_4 = [self.backbone2.layer4(chunk_event_3[i]) for i in range(self.K)]
        chunk_event_5 = [self.backbone2.deconv_layer.deconv_layers(chunk_event_4[i]) for i in range(self.K)]
        
        ## fuison
        chunk_0 = [self.fus[0](chunk_image_0[i], chunk_event_0[i]) for i in range(self.K)]
        
        chunk_1 = [self.fus[1](chunk_image_1[i], chunk_event_1[i]) for i in range(self.K)]
        chunk_1 = [torch.cat((chunk_1[i], chunk_0[i]), dim=1) for i in range(self.K)]
        chunk_1 = [self.middle_fusion_1_ful(chunk_1[i]) for i in range(self.K)]
        
        chunk_2 = [self.fus[2](chunk_image_2[i], chunk_event_2[i]) for i in range(self.K)]
        chunk_2 = [torch.cat((chunk_2[i], self.maxpool(chunk_1[i])), dim=1) for i in range(self.K)]
        chunk_2 = [self.middle_fusion_2_ful(chunk_2[i]) for i in range(self.K)]
        
        chunk_3 = [self.fus[3](chunk_image_3[i], chunk_event_3[i]) for i in range(self.K)]
        chunk_3 = [torch.cat((chunk_3[i], self.maxpool(chunk_2[i])), dim=1) for i in range(self.K)]
        chunk_3 = [self.middle_fusion_3_ful(chunk_3[i]) for i in range(self.K)]
        
        chunk_4 = [self.fus[4](chunk_image_4[i], chunk_event_4[i]) for i in range(self.K)]
        chunk_4 = [torch.cat((chunk_4[i], self.maxpool(chunk_3[i])), dim=1) for i in range(self.K)]
        chunk_4 = [self.middle_fusion_4_ful(chunk_4[i]) for i in range(self.K)]
        
        chunk_5 = [self.fus[5](chunk_image_5[i], chunk_event_5[i]) for i in range(self.K)]
        chunk_5 = [torch.cat((chunk_5[i], self.deconv_layer_2(chunk_4[i])), dim=1) for i in range(self.K)]
        chunk_5 = [self.middle_fusion_5_ful(chunk_5[i]) for i in range(self.K)]

        chunk = chunk_5
        
        return [self.branch(chunk)]
