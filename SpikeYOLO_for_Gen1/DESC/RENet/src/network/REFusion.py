import torch
import torch.nn as nn

from .Attentions import *


class REFusion(nn.Module):
	def __init__(self, in_planes, out_planes, layer):
		super(REFusion, self).__init__()
		self.ChannelGate_rgb = ChannelGate(out_planes, 16)
		self.ChannelGate_evt = ChannelGate(out_planes, 16)
		self.conv = nn.Sequential(nn.Conv2d(out_planes + out_planes, out_planes, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_planes), nn.ReLU(inplace=True))
		self.SpatialGate = SpatialGate()
		if (layer != 0) & (layer != 5):
			self.projection = nn.Sequential(nn.Conv2d(out_planes // 4, out_planes, kernel_size=3, stride=1, padding=1, bias=True), nn.BatchNorm2d(out_planes, momentum=0.1), nn.ReLU(inplace=True))
		self.layer = layer
		self.conv0_rgb = nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=1, padding=0)
		self.conv0_evt = nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=1, padding=0)
		self.conv1_rgb = nn.Sequential(nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_planes), nn.ReLU(inplace=True))
		self.conv1_evt = nn.Sequential(nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_planes), nn.ReLU(inplace=True))

	def forward(self, rgb, evt):
		if (self.layer != 0) & (self.layer != 5):
			evt = self.projection(evt)

		rgb0 = self.conv0_rgb(rgb)
		evt0 = self.conv0_evt(evt)

		mul = rgb0.mul(evt0)

		rgb_mul = rgb0 + mul
		evt_mul = evt0 + mul

		rgb_chn_att = self.ChannelGate_rgb(rgb_mul)
		evt_chn_att = self.ChannelGate_evt(evt_mul)

		rgb_crs_att = rgb_mul * evt_chn_att
		evt_crs_att = evt_mul * rgb_chn_att

		rgb1 = rgb_mul + rgb_crs_att
		evt1 = evt_mul + evt_crs_att

		rgb_spt_att = self.SpatialGate(rgb1)
		evt_spt_att = self.SpatialGate(evt1)

		rgb_crs_att_2 = rgb1 * evt_spt_att
		evt_crs_att_2 = evt1 * rgb_spt_att

		rgb1_2 = rgb1 + rgb_crs_att_2
		evt1_2 = evt1 + evt_crs_att_2

		rgb2 = self.conv1_rgb(rgb1_2)
		evt2 = self.conv1_evt(evt1_2)

		mul2 = torch.mul(rgb2, evt2)

		max_rgb = torch.reshape(rgb2,[rgb2.shape[0],1,rgb2.shape[1],rgb2.shape[2],rgb2.shape[3]])
		max_evt = torch.reshape(evt2,[evt2.shape[0],1,evt2.shape[1],evt2.shape[2],evt2.shape[3]])
		max_cat = torch.cat((max_rgb, max_evt), dim=1)
		max_out = max_cat.max(dim=1)[0]

		out_mul_max = torch.cat((mul2, max_out), dim=1)

		out = out_mul_max

		out = self.conv(out)

		return out
