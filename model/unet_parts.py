""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

class Down_new(nn.Module):
	"""Downscaling with maxpool then double conv"""

	def __init__(self, in_channels, out_channels, lrelu=False):
		super().__init__()
		if lrelu:
			activation = nn.LeakyReLU()
		else:
			activation = nn.ReLU()
		self.down_conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=4, padding=1, stride=2),
			nn.BatchNorm2d(out_channels),
			activation
		)

	def forward(self, x):
		return self.down_conv(x)

class Up_new(nn.Module):
	"""Upscaling then double conv"""

	def __init__(self, in_channels, out_channels, bilinear=True, dropout_rate=0.0, lrelu=False):
		super().__init__()

		# if bilinear, use the normal convolutions to reduce the number of channels
		if bilinear:
			self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
			if lrelu:
				activation = nn.LeakyReLU()
			else:
				activation = nn.ReLU()
			self.conv = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
				nn.BatchNorm2d(out_channels),
				nn.Dropout2d(p=dropout_rate),
				activation # nn.ReLU(inplace=True)
			)
		else:
			# self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
			# self.conv = DoubleConv(in_channels, out_channels)
			raise NotImplementedError

	def forward(self, x1, x2):
		x1 = self.up(x1)
		x1 = self.conv(x1)
		return torch.cat([x2,x1], dim=1)

class Up_solo(Up_new):
	"""Upscaling then double conv"""
	def __init__(self, in_channels, out_channels, bilinear=True, dropout_rate=0.0, lrelu=False):
		super().__init__(in_channels, out_channels, bilinear=bilinear, dropout_rate=dropout_rate, lrelu=lrelu)

	def forward(self, x1):
		x1 = self.up(x1)
		x1 = self.conv(x1)
		return x1

class OutConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(OutConv, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

	def forward(self, x):
		return self.conv(x)

class OutConv_new(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(OutConv_new, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
			nn.Tanh()
		)
	def forward(self, x):
		return self.conv(x)

class Down_refine(nn.Module):
	"""Downscaling with maxpool then double conv"""

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.down_conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=4, padding=1, stride=2),
			nn.BatchNorm2d(out_channels),
			nn.LeakyReLU()
		)

	def forward(self, x):
		return self.down_conv(x)