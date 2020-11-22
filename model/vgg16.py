import torch
import torch.nn as nn
from torchvision.models import vgg16_bn, vgg16
from collections import namedtuple
import os

# reference: https://discuss.pytorch.org/t/accessing-intermediate-layers-of-a-pretrained-network-forward/12113

class Vgg16(torch.nn.Module):
	def __init__(self):
		super(Vgg16, self).__init__()
		os.environ['TORCH_HOME'] = 'model'
		features = list(vgg16(pretrained = False).features)
		# features = torch.load("model/checkpoints/modified_vgg.pth")["model"]
		# features = torch.load("model/checkpoints/vgg16-397923af.pth")
		self.mean = torch.Tensor([123.68,116.779,103.939]).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		self.mean = self.mean.reshape((1,3,1,1))
		self.features = nn.ModuleList(features).eval() 
		
	def forward(self, x):
		results = []
		
		x = x - self.mean 
		for ii,model in enumerate(self.features):

			x = model(x)
			if ii in {1,8,15}: # relu1_1, relu2_2, relu3_3
				results.append(x)
			if ii == 15:
				break
		return results