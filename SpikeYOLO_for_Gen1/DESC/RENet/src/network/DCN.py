from torchvision.ops.deform_conv import *
import torch
import torch.nn as nn

class DCN(nn.Module):
    def __init__(self, inc, out, kernel_size, stride, padding, dilation=1, deformable_groups=1):
        super(DCN, self).__init__()
        self.conv_offset_mask = nn.Conv2d(inc, 3*kernel_size*kernel_size, 3, padding=1)
        self.init_offset()
        self.weight = nn.Parameter(torch.empty(out, inc, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out))

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, x):
        out = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return deform_conv2d(x, offset, self.weight, self.bias, padding=1, mask=mask)
