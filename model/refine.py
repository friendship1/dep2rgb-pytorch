import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_parts import *


class RefineNet(nn.Module):
	def __init__(self, n_channels, bilinear=True):
		super().__init__()
		self.n_channels = n_channels
		self.bilinear = bilinear
		temp_lrelu = False
		self.down0 = Down_new(n_channels, 256, lrelu=temp_lrelu)
		self.down1 = Down_new(256, 256, lrelu=temp_lrelu)
		self.down2 = Down_new(256, 256, lrelu=temp_lrelu)
		self.down3 = Down_new(256, 512, lrelu=temp_lrelu)
		self.down4 = Down_new(512, 512, lrelu=temp_lrelu)
		self.down5 = Down_new(512, 512, lrelu=temp_lrelu)
		# factor = 2 if bilinear else 1
		self.up1 = Up_new(512, 512, bilinear, 0.5, lrelu=True)
		self.up2 = Up_new(1024, 512, bilinear, 0.5, lrelu=True)
		self.up3 = Up_new(1024, 512, bilinear, 0.5, lrelu=True)
		self.up4 = Up_new(768, 256, bilinear, lrelu=True)

		self.up5 = Up_solo(512, 256, bilinear, lrelu=True) # Up_new(512, 256, bilinear, lrelu=True)
		self.up6 = Up_solo(256, 256, bilinear, lrelu=True)
		self.last = nn.Sequential(
			nn.Conv2d(256, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(),
			nn.Conv2d(128, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(),
			nn.Conv2d(64, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(),
			nn.Conv2d(32, 3, kernel_size=3, padding=1),
			nn.Tanh()
		)

	def forward(self, x):
		x1 = self.down0(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x6 = self.down5(x5)
		u5 = self.up1(x6, x5)
		u4 = self.up2(u5, x4)
		u3 = self.up3(u4, x3)
		u2 = self.up4(u3, x2)
		u1 = self.up5(u2)
		u0 = self.up6(u1)
		out = self.last(u0)
		# logits = self.last(out)
		return out

class DiscriNet(nn.Module):
	def __init__(self, in_channels):
		super().__init__()
		self.down0 = Down(in_channels, 256)
		# self.concat = torch.cat([x2,x1], dim=1)
		self.down1 = Down(384, 256)
		self.down2 = Down(512, 256)
		self.down3 = Down(256, 512)
		self.down4 = Down(512, 512)
		self.down5 = Down(512, 512)
		self.last = nn.Sequential(
			# nn.Linear(26624 * 4, 1024),
			nn.Linear(26624, 1024),
			nn.Dropout(p=0.5),
			nn.LeakyReLU(),

			nn.Linear(1024, 1024),
			nn.Dropout(p=0.5),
			nn.LeakyReLU(),

			nn.Linear(1024, 1024),
			nn.Dropout(p=0.5),
			nn.LeakyReLU(),

			nn.Linear(1024, 2),
		)

	def forward(self, inp):
		x0 = self.down0(inp[0])
		x0 = torch.cat((x0, inp[1]), dim=1)
		x1 = self.down1(x0)
		x1 = torch.cat((x1, inp[2]), dim=1)
		x2 = self.down2(x1)
		
		x = self.down3(x2)
		x = self.down4(x)
		x = self.down5(x)
		x = x.reshape(x.size(0), -1)
		x = self.last(x)
		return x

class Down(nn.Module):
	"""Downscaling with maxpool then double conv"""

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.down_conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
			nn.BatchNorm2d(out_channels),
			nn.LeakyReLU(),
			nn.MaxPool2d(2)
		)

	def forward(self, x):
		return self.down_conv(x)