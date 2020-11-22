""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
from .unet_parts import *

class UNet(nn.Module):
	def __init__(self, n_channels, n_classes, bilinear=True):
		super(UNet, self).__init__()
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.bilinear = bilinear

		self.down0 = Down_new(n_channels, 256)
		self.down1 = Down_new(256, 256)
		self.down2 = Down_new(256, 256)
		self.down3 = Down_new(256, 512)
		self.down4 = Down_new(512, 512)
		self.down5 = Down_new(512, 512)
		# factor = 2 if bilinear else 1
		self.up1 = Up_new(512, 512, bilinear, 0.5)
		self.up2 = Up_new(1024, 512, bilinear, 0.5)
		self.up3 = Up_new(1024, 512, bilinear, 0.5)
		self.up4 = Up_new(768, 256, bilinear)
		self.up5 = Up_new(512, 256, bilinear)
		self.up6 = Up_new(512, 256, bilinear)
		self.last = nn.Sequential(
			nn.Conv2d(256 + n_channels, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
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
		u1 = self.up5(u2, x1)
		u0 = self.up6(u1, x)
		out = self.last(u0)
		# logits = self.last(out)
		return out