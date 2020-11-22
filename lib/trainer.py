# -*- coding: future_fstrings -*-
#
# Written by Chris Choy <chrischoy@ai.stanford.edu>
# Distributed under MIT License
import os
import os.path as osp
import gc
import logging
import numpy as np
import json

import torch
import torch.optim as optim
import torch.nn.functional as F
# import torchvision.models as tvmodels

from tensorboardX import SummaryWriter
from torchsummary import summary

import torchvision
import torchvision.utils 

from model import load_model
# import util.transform_estimation as te
from lib.metrics import pdist, corr_dist
from lib.timer import Timer, AverageMeter
from lib.eval import find_nn_gpu

from util.file import ensure_dir
from util.misc import _hash

import MinkowskiEngine as ME


class BaseTrainer:

	def __init__(
		self,
		config,
		data_loader,
		val_data_loader=None,
	):
		num_feats = 1  # occupancy only for 3D Match dataset. For ScanNet, use RGB 3 channels.

		# Model initialization
		Model_3dnet = load_model(config.model_3dnet)
		model_3dnet = Model_3dnet(
			num_feats, # in_channels
			config.model_n_out,
			bn_momentum=config.bn_momentum,
			normalize_feature=config.normalize_feature,
			conv1_kernel_size=config.conv1_kernel_size,
			D=3)
		
		# this is for unet
		Coarsenet = load_model(config.coarsenet)
		coarsenet = Coarsenet(
			1, #config.model_n_out,
			3)

		Model_VGG = load_model("Vgg16")
		model_vgg = Model_VGG()
		state = torch.load("model/checkpoints/modified_vgg.pth")["model"]
		print(state.keys())
		model_vgg.features.load_state_dict(state)
		'''
		logging.info(model_3dnet)
		logging.info(coarsenet)

		print("parameters in model_3dnet :")
		model_parameters = filter(lambda p: p.requires_grad, model_3dnet.parameters())
		params = sum([np.prod(p.size()) for p in model_parameters])
		print(params)

		print("parameters in coarsenet :")
		model_parameters = filter(lambda p: p.requires_grad, coarsenet.parameters())
		params = sum([np.prod(p.size()) for p in model_parameters])
		print(params)
		'''
		self.config = config
		self.model_3dnet = model_3dnet
		self.coarsenet = coarsenet
		self.model_vgg = model_vgg
		self.max_epoch = config.max_epoch
		self.val_max_iter = config.val_max_iter
		self.val_epoch_freq = config.val_epoch_freq

		self.best_val_metric = config.best_val_metric
		self.best_val_epoch = -np.inf
		self.best_val = -np.inf

		self.epoch_cnt = 0

		if config.use_gpu and not torch.cuda.is_available():
			logging.warning('Warning: There\'s no CUDA support on this machine, '
							'training is performed on CPU.')
			raise ValueError('GPU not available, but cuda flag set')

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		# summary for debug by wjw
		# model.cuda()
		# summary(model, (3,6000))
		# coarsenet.cuda()
		#summary(coarsenet, (32,384,1248))

		self.optimizer = getattr(optim, config.optimizer)(
			coarsenet.parameters(),
			lr=config.lr)
			# eps=1e-8,
			# momentum=config.momentum,
			# weight_decay=config.weight_decay)

		self.start_epoch = 1
		self.checkpoint_dir = config.out_dir
		self.rgbs_dir = config.rgbs_dir

		ensure_dir(self.checkpoint_dir)
		# write config.json in outdir
		json.dump(
			config,
			open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
			indent=4,
			sort_keys=False)

		self.iter_size = config.iter_size
		self.batch_size = data_loader.batch_size
		self.data_loader = data_loader
		self.val_data_loader = val_data_loader

		self.test_valid = True if self.val_data_loader is not None else False
		self.log_step = int(np.sqrt(self.config.batch_size))
		self.model_3dnet = self.model_3dnet.to(self.device)
		self.coarsenet = self.coarsenet.to(self.device)
		self.model_vgg = self.model_vgg.to(self.device)
		self.writer = SummaryWriter(logdir=config.out_dir)

		if config.resume is not None:
			if osp.isfile(config.resume):
				logging.info("=> loading checkpoint '{}'".format(config.resume))
				state = torch.load(config.resume)
				self.start_epoch = state['epoch'] + 1
				# model_3dnet.load_state_dict(state['state_dict'])
				coarsenet.load_state_dict(state['state_dict2'])
				self.optimizer.load_state_dict(state['optimizer'])

				if 'best_val' in state.keys():
					self.best_val = state['best_val']
					self.best_val_epoch = state['best_val_epoch']
					self.best_val_metric = state['best_val_metric']
			else:
				raise ValueError(f"=> no checkpoint found at '{config.resume}'")

	def train(self):
		"""
		Full training logic
		"""
		self._save_checkpoint(0)
		# Baseline random feature performance
		if self.test_valid:
			val_dict = self._valid_epoch()
			for k, v in val_dict.items():
				self.writer.add_scalar(f'val/{k}', v, 0)

		for epoch in range(self.start_epoch, self.max_epoch + 1):
			self.epoch_cnt = epoch
			logging.info(f" Epoch: {epoch}, LR: None")
			self._train_epoch(epoch)
			self._save_checkpoint(epoch)

			if self.test_valid and epoch % self.val_epoch_freq == 0:
				val_dict = self._valid_epoch()
				for k, v in val_dict.items():
					self.writer.add_scalar(f'val/{k}', v, epoch)
				if self.best_val < val_dict[self.best_val_metric]:
					logging.info(
						f'Saving the best val model with {self.best_val_metric}: {val_dict[self.best_val_metric]}'
					)
					self.best_val = val_dict[self.best_val_metric]
					self.best_val_epoch = epoch
					self._save_checkpoint(epoch, 'best_val_checkpoint')
				else:
					logging.info(
						f'Current best val model with {self.best_val_metric}: {self.best_val} at epoch {self.best_val_epoch}'
					)

	def test(self):
		tot_num_data = len(self.val_data_loader.dataset)
		self.val_max_iter = tot_num_data
		print("Test Starts")
		self._valid_epoch(self.config.is_test) # need args
		print("Test Ends")

	def _save_checkpoint(self, epoch, filename='checkpoint'):
		state = {
			'epoch': epoch,
			# 'state_dict': self.model_3dnet.state_dict(),
			'state_dict2': self.coarsenet.state_dict(),
			'optimizer': self.optimizer.state_dict(),
			'config': self.config,
			'best_val': self.best_val,
			'best_val_epoch': self.best_val_epoch,
			'best_val_metric': self.best_val_metric
		}
		filename = os.path.join(self.checkpoint_dir, f'{filename+str(epoch).zfill(4)}.pth')
		# filename = os.path.join(self.checkpoint_dir, f'{filename}.pth')
		logging.info("Saving checkpoint: {} ...".format(filename))
		torch.save(state, filename)


class BasicTrainer(BaseTrainer):

	def __init__(
			self,
			config,
			data_loader,
			val_data_loader=None,
		):
		if val_data_loader is not None:
			assert val_data_loader.batch_size == 1, "Val set batch size must be 1 for now."
		BaseTrainer.__init__(self, config, data_loader, val_data_loader)
		self.perceptual_loss_coff = 1
	
	def project2d(self, batch_size, F, coord2d):
		batch_size = coord2d.shape[0]
		# print(coord2d.shape)
		ret = ME.SparseTensor(
			F.F,
			coords = coord2d
		)
		ret_dense, min_coords, stride = ret.dense() # (max_coords=torch.tensor([383,1247]).int(), min_coords=torch.tensor([0,0]).int())        
		pad_size_x = 1248 - ret_dense.shape[2]
		pad_size_y = 384 - ret_dense.shape[3]

		ret_dense_pad = torch.nn.functional.pad(input=ret_dense, pad=(pad_size_y,0,pad_size_x,0), mode='constant', value=0)
		ret_dense_pad = ret_dense_pad.permute(0,1,3,2)
		return ret_dense_pad

	def perceptual_loss(self, syn, ori):
		syn = syn
		p1, p2, p3 = self.model_vgg(syn)
		g1, g2, g3 = self.model_vgg(ori)
		# loss = torch.nn.MSELoss()
		# return ((loss(p1,g1) + loss(p2,g2) + loss(p3,g3)) / 3) * self.perceptual_loss_coff
		return ( torch.mean((p1-g1)**2) + torch.mean((p2-g2)**2) + torch.mean((p3-g3)**2) ) / 3.0

	def _train_epoch(self, epoch):
		gc.collect()
		self.model_3dnet.train()
		self.coarsenet.train()
		self.model_vgg.eval()
		# Epoch starts from 1

		data_loader = self.data_loader
		data_loader_iter = self.data_loader.__iter__()

		iter_size = self.iter_size
		start_iter = (epoch - 1) * (len(data_loader))

		data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()
		batch_loss = 0
		batch_percep_loss = 0
		batch_recon_loss = 0

		# Main training
		for curr_iter in range(len(data_loader)):
			self.optimizer.zero_grad()

			data_time = 0
			total_timer.tic()
			data_timer.tic()
			input_dict = data_loader_iter.next()
			data_time += data_timer.toc(average=False)

			rgb_GT = input_dict['img'].to(self.device)
			F2d = input_dict['dep'].to(self.device)
			if(self.config.allone == 1): # Sort of abliation study (will be removed)
				F2d = F2d.cpu().numpy()
				F2d = np.where(F2d != 0, 1, 0)
				F2d = torch.from_numpy(F2d).float().to(self.device)
			RGB = self.coarsenet(F2d)
			RGB = (RGB + 1) * 127.5

			recon_loss = torch.mean(torch.abs(RGB - rgb_GT))
			percep_loss = self.perceptual_loss(RGB, rgb_GT)
			loss = percep_loss + recon_loss

			loss.backward()  # To accumulate gradient, zero gradients only at the begining of iter_size

			batch_size = rgb_GT.shape[0]
			bchavg_loss = loss.item() / batch_size
			bchavg_percep_loss = percep_loss.item() / batch_size
			bchavg_recon_loss = recon_loss.item() / batch_size

			self.optimizer.step()

			torch.cuda.empty_cache()
			total_timer.toc()
			data_meter.update(data_time)

			# Print logs
			if curr_iter % self.config.stat_freq == 0:
				# print(torch.min(rgb_GT), torch.max(rgb_GT))
				rgb_GT_print = torch.round(rgb_GT) # torch.round((rgb_GT + 1) * 127.5)
				RGB_print = torch.round(RGB) # torch.round((RGB + 1) * 127.5)
				F2d_print = torch.round((F2d / torch.max(F2d)) * 255.0)
				F2d_print = torch.cat((F2d_print,F2d_print,F2d_print), dim=1)
				Board_image = torch.cat((rgb_GT_print, RGB_print, F2d_print), dim=2).type(torch.uint8)
				# print(Board_image.shape)
				self.writer.add_scalar('train/loss', bchavg_loss, start_iter + curr_iter)
				self.writer.add_scalar('train/percep_loss', bchavg_percep_loss, start_iter + curr_iter)
				self.writer.add_scalar('train/recon_loss', bchavg_recon_loss, start_iter + curr_iter)
				grid_img = torchvision.utils.make_grid(Board_image)
				# self.writer.add_image('images/image',grid_img,start_iter + curr_iter)
				logging.info(
					"Train Epoch: {} [{}/{}], Current Loss: {:.6f} per.loss: {:.6f} recon.loss: {:.6f} "
					.format(epoch, curr_iter,    
							len(self.data_loader)
							, bchavg_loss, bchavg_percep_loss, bchavg_recon_loss) +
					"\tData time: {:.4f}, Train time: {:.4f}".format(
						data_meter.avg, total_timer.avg - data_meter.avg))
				data_meter.reset()
				total_timer.reset()

	def _valid_epoch(self, is_test=False):
		self.model_3dnet.eval()
		self.coarsenet.eval()
		self.model_vgg.eval()
		self.val_data_loader.dataset.reset_seed(0)
		num_data = 0
		loss_meter = AverageMeter()
		data_timer, feat_timer = Timer(), Timer()
		tot_num_data = len(self.val_data_loader.dataset)
		if self.val_max_iter > 0:
			tot_num_data = min(self.val_max_iter, tot_num_data)
		data_loader_iter = self.val_data_loader.__iter__()

		for batch_idx in range(tot_num_data):
			data_timer.tic()
			input_dict = data_loader_iter.next()
			data_timer.toc()

			feat_timer.tic()
			rgb_GT = input_dict['img'].to(self.device)
			F2d = input_dict['dep'].to(self.device)
			if(self.config.allone == 1): # Sort of abliation study (will be removed)
				F2d = F2d.cpu().numpy()
				F2d = np.where(F2d != 0, 1, 0)
				F2d = torch.from_numpy(F2d).float().to(self.device)
			RGB = self.coarsenet(F2d)
			RGB = (RGB + 1) * 127.5
			# RGB = F.tanh(RGB)
			
			feat_timer.toc()
			recon_loss = torch.mean(torch.abs(RGB - rgb_GT))
			percep_loss = self.perceptual_loss(RGB, rgb_GT)
			loss = percep_loss + recon_loss

			# save RGB
			# rgb_filename = os.path.join(self.rgbs_dir, f'{"rbgv1_"+str(self.epoch_cnt).zfill(4)+"_"+str(batch_idx).zfill(3)}.pth')
			# torch.save(RGB, rgb_filename)
			loss_meter.update(loss.item())

			num_data += 1
			torch.cuda.empty_cache()

			if is_test:
				import cv2
				rgb_GT_print = torch.round(rgb_GT) # torch.round((rgb_GT + 1) * 127.5)
				RGB_print = torch.round(RGB) # torch.round((RGB + 1) * 127.5)
				F2d_print = torch.round((F2d / torch.max(F2d)) * 255.0)
				F2d_print = torch.cat((F2d_print,F2d_print,F2d_print), dim=1)
				Board_image = torch.cat((rgb_GT_print, RGB_print, F2d_print), dim=2).type(torch.uint8)
				Board_image = torch.squeeze(Board_image)
				# grid_img = torchvision.utils.make_grid(Board_image)
				# self.writer.add_image('images/image_val',grid_img, batch_idx)
				Board_image_np = Board_image.cpu().numpy()[::-1,:,:] # reverse color channel order
				Board_image_np = np.transpose(Board_image_np, (1, 2, 0)) # Becasue of cv2 GBR color change
				# print(Board_image_np.shape)
				rgb_filename = os.path.join(self.rgbs_dir, f'{"rbgv1_"+str(self.start_epoch).zfill(4)+"_"+str(batch_idx).zfill(3)}.png')
				cv2.imwrite(rgb_filename, Board_image_np)

			if batch_idx % 100 == 0 and batch_idx > 0:
				logging.info(' '.join([
					f"Validation iter {num_data} / {tot_num_data} : Data Loading Time: {data_timer.avg:.3f},",
					f"Loss: {loss_meter.avg:.6f}, ",
				]))
				data_timer.reset()

		logging.info(' '.join([
			f"Final Loss: {loss_meter.avg:.6f}, "
		]))
		return {
			"loss": loss_meter.avg,
		}