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
		# num_feats = 1  # occupancy only for 3D Match dataset. For ScanNet, use RGB 3 channels.

		# Model initialization
		# this is for unet
		Coarsenet = load_model(config.coarsenet)
		coarsenet = Coarsenet(
			1, #config.model_n_out,
			3)

		# VGG
		Model_VGG = load_model("Vgg16")
		model_vgg = Model_VGG()
		state = torch.load("model/checkpoints/modified_vgg.pth")["model"]
		# print(state.keys())
		model_vgg.features.load_state_dict(state)

		# GAN
		Refinenet = load_model(config.refinenet)
		refinenet = Refinenet(4) # cpred(3) + cinp(1)

		Discrinet = load_model(config.discrinet)
		discrinet = Discrinet(71) # in_channels

		# logging.info(model_3dnet)
		# logging.info(coarsenet)
		# logging.info(model_vgg)
		# logging.info(refinenet)
		# logging.info(discrinet)

		# print("parameters in coarsenet :")
		# model_parameters = filter(lambda p: p.requires_grad, coarsenet.parameters())
		# params = sum([np.prod(p.size()) for p in model_parameters])
		# print(params)

		self.config = config
		self.coarsenet = coarsenet
		self.model_vgg = model_vgg
		self.refinenet = refinenet
		self.discrinet = discrinet

		self.CEloss = torch.nn.CrossEntropyLoss()

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
		self.device0 = torch.device('cuda:0')
		self.device1 = torch.device('cuda:0')

		# self.optimizer = getattr(optim, config.optimizer)(
		# 	coarsenet.parameters(),
		# 	lr=config.lr)
		# 	# eps=1e-8,
		# 	# momentum=config.momentum,
		# 	# weight_decay=config.weight_decay)

		self.optimizer_refine = getattr(optim, config.optimizer)(
			self.refinenet.parameters(),
			lr=config.lr)
		self.optimizer_discri = getattr(optim, config.optimizer)(
			self.discrinet.parameters(),
			lr=config.lr)

		self.start_epoch = 1
		self.checkpoint_dir = config.out_dir
		self.rgbs_dir = config.rgbs_dir

		ensure_dir(self.checkpoint_dir)
		# write config.json in output dir
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

		self.coarsenet = self.coarsenet.to(self.device0)
		self.model_vgg = self.model_vgg.to(self.device0)
		self.refinenet = self.refinenet.to(self.device0)
		self.discrinet = self.discrinet.to(self.device1)

		self.writer = SummaryWriter(logdir=config.out_dir)

		if os.path.exists(config.coarsenet_pth):
			state = torch.load(config.coarsenet_pth)
			coarsenet.load_state_dict(state['state_dict2'])
			# self.optimizer.load_state_dict(state['optimizer'])
		else:
			raise OSError(f"{config.coarsenet_pth} does not exist")

		if config.resume is not None:
			if osp.isfile(config.resume):
				logging.info("=> loading checkpoint '{}'".format(config.resume))
				state = torch.load(config.resume)
				self.start_epoch = state['epoch'] + 1
				refinenet.load_state_dict(state['state_dict_refine'])
				discrinet.load_state_dict(state['state_dict_discri'])
				self.optimizer_discri.load_state_dict(state['optimizer_discri'])
				self.optimizer_refine.load_state_dict(state['optimizer_refine'])

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
			'state_dict_refine': self.refinenet.state_dict(),
			'state_dict_discri': self.discrinet.state_dict(),
			'optimizer_discri': self.optimizer_discri.state_dict(),
			'optimizer_refine': self.optimizer_refine.state_dict(),
			'config': self.config,
			'best_val': self.best_val,
			'best_val_epoch': self.best_val_epoch,
			'best_val_metric': self.best_val_metric
		}
		filename = os.path.join(self.checkpoint_dir, f'{filename+str(epoch).zfill(4)}.pth')
		# filename = os.path.join(self.checkpoint_dir, f'{filename}.pth')
		logging.info("Saving checkpoint: {} ...".format(filename))
		torch.save(state, filename)


class RefineTrainer(BaseTrainer):
	def __init__(
			self,
			config,
			data_loader,
			val_data_loader=None,
		):
		if val_data_loader is not None:
			assert val_data_loader.batch_size == 1, "Val set batch size must be 1 for now."
		BaseTrainer.__init__(self, config, data_loader, val_data_loader)
	
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
		self.coarsenet.eval()
		self.model_vgg.eval()
		self.refinenet.train()
		self.discrinet.train()

		
		dloss_prev = 1e6
		# Epoch starts from 1

		data_loader = self.data_loader
		data_loader_iter = self.data_loader.__iter__()

		iter_size = self.iter_size
		start_iter = (epoch - 1) * (len(data_loader))

		data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()
		batch_loss = 0
		batch_recon_loss = 0
		train_discri_cnt = 0
		train_discri_candi_cnt = 0
		discri_loss_avg = 0
		refine_loss_avg = 0
		recon_loss_avg = 0
		percep_loss_avg = 0
		ce_loss_avg = 0

		# Main training
		# first_batch = next(iter(data_loader))
		# for curr_iter, input_dict in enumerate([first_batch] * 10000):
    	# training code here

		for curr_iter in range(len(data_loader)):
			# self.optimizer.zero_grad()
			self.optimizer_refine.zero_grad()
			self.optimizer_discri.zero_grad()
			self.model_vgg.zero_grad()

			data_time = 0
			total_timer.tic()
			data_timer.tic()
			input_dict = data_loader_iter.next()
			data_time += data_timer.toc(average=False)

			rgb_gt = input_dict['img'].to(self.device0)
			F2d = input_dict['dep'].to(self.device0)
			with torch.no_grad():
				rgb = self.coarsenet(F2d)
			rgb_255 = (rgb + 1.0) * 127.5

			batch_size = rgb_gt.shape[0]
			label_real = torch.ones(batch_size, dtype=torch.long).to(self.device1)
			label_real0 = torch.ones(batch_size, dtype=torch.long).to(self.device0)
			label_fake = torch.zeros(batch_size, dtype=torch.long).to(self.device1)

			# RefineNet
			refine_inp = torch.cat((rgb_255, F2d), dim=1) # following invsfm
			rgb_refine = self.refinenet(refine_inp)
			rgb_refine = (rgb_refine + 1.0) * 127.5

			### Adversarial Training
			if dloss_prev > 0.1:
				train_discri_candi_cnt+=1
			# if curr_iter % 2 == 0 and dloss_prev > 0.1:
			if dloss_prev > 0.1:
				self.refinenet.eval()
				self.discrinet.train()
				train_discri = 1
				train_discri_cnt+=1
			else:
				self.refinenet.train()
				self.discrinet.eval()
				train_discri = 0

			# DiscriNet
			g1, g2, g3 = self.model_vgg(rgb_gt)
			real_inp = [torch.cat((refine_inp, rgb_gt, g1), dim=1).to(self.device1).detach(), g2.to(self.device1).detach(), g3.to(self.device1).detach()]
			### Discriminator loss

			###################################################################### BUGs HERE MAYBE
			# real backward
			gout = self.discrinet(real_inp)
			discri_loss_real = self.CEloss(gout, label_real)
			# if train_discri:
			# 	discri_loss_real.backward()
			# fake backward
			f1, f2, f3 = self.model_vgg(rgb_refine)
			fake_inp = [torch.cat((refine_inp, rgb_refine, f1), dim=1).to(self.device1), f2.to(self.device1), f3.to(self.device1)]
			fout = self.discrinet(fake_inp)
			discri_loss_fake = self.CEloss(fout, label_fake)
			# if train_discri:
			# 	discri_loss_fake.backward()
			discri_loss = discri_loss_real + discri_loss_fake
			if train_discri:
				discri_loss.backward()

			# RefineNet loss (Generator)
			recon_loss = torch.mean(torch.abs(rgb_refine - rgb_gt))
			percep_loss = ( torch.mean((f1-g1)**2) + torch.mean((f2-g2)**2) + torch.mean((f3-g3)**2) ) / 3.0
			ce_loss = self.CEloss(fout.to(self.device0), label_real0)
			refine_loss = percep_loss + recon_loss + 1e4 * ce_loss
			
			### Adversarial Training 
			if train_discri:
				# discri_loss.backward()
				self.optimizer_discri.step()
			else:
				refine_loss.backward()
				self.optimizer_refine.step()
			dloss_prev = discri_loss.item()
			#####################################################################
			bchavg_discri_loss = dloss_prev / batch_size
			bchavg_refine_loss = refine_loss.item() / batch_size
			bchavg_recon_loss = recon_loss.item() / batch_size
			bchavg_percep_loss = percep_loss.item() / batch_size
			bchavg_ce_loss = ce_loss.item() / batch_size
			# self.optimizer.step()

			discri_loss_avg += bchavg_discri_loss
			refine_loss_avg += bchavg_refine_loss
			recon_loss_avg += bchavg_recon_loss
			percep_loss_avg += bchavg_percep_loss
			ce_loss_avg += bchavg_ce_loss

			torch.cuda.empty_cache()
			total_timer.toc()
			data_meter.update(data_time)


			# Print logs
			if curr_iter % self.config.stat_freq == 0:
				discri_loss_avg /= self.config.stat_freq
				refine_loss_avg /= self.config.stat_freq
				recon_loss_avg /= self.config.stat_freq
				percep_loss_avg /= self.config.stat_freq
				ce_loss_avg /= self.config.stat_freq
				# rgb_gt_print = torch.round(rgb_gt) # torch.round((rgb_gt + 1) * 127.5)
				# rgb_print = torch.round(rgb) # torch.round((rgb + 1) * 127.5)
				# F2d_print = torch.round((F2d / torch.max(F2d)) * 255.0)
				# F2d_print = torch.cat((F2d_print,F2d_print,F2d_print), dim=1)
				# Board_image = torch.cat((rgb_gt_print, rgb_print, F2d_print), dim=2).type(torch.uint8)
				# print(Board_image.shape)
				self.writer.add_scalar('train/discri_loss', discri_loss_avg, start_iter + curr_iter)
				self.writer.add_scalar('train/refine_loss', refine_loss_avg, start_iter + curr_iter)
				self.writer.add_scalar('train/recon_loss', bchavg_recon_loss, start_iter + curr_iter)
				self.writer.add_scalar('train/percep_loss', bchavg_percep_loss, start_iter + curr_iter)
				self.writer.add_scalar('train/ce_loss', bchavg_ce_loss, start_iter + curr_iter)
				# grid_img = torchvision.utils.make_grid(Board_image)
				# self.writer.add_image('images/image',grid_img,start_iter + curr_iter)
				logging.info(
					"Train Epoch: {} [{}/{}], Current discri.loss: {:.6f} refine.loss: {:.6f} Dtrn_cnt:{}/{} recon: {:.5f} percep: {:.5f} ce: {:.5f}"
					.format(epoch, curr_iter, len(self.data_loader)
							, discri_loss_avg, refine_loss_avg, train_discri_cnt, train_discri_candi_cnt, recon_loss_avg, percep_loss_avg, ce_loss_avg) +
					"\tData time: {:.4f}, Train time: {:.4f}".format(
						data_meter.avg, total_timer.avg - data_meter.avg))
				data_meter.reset()
				total_timer.reset()
				discri_loss_avg = 0
				refine_loss_avg = 0
				train_discri_cnt = 0
				train_discri_candi_cnt = 0
				recon_loss_avg = 0
				percep_loss_avg = 0
				ce_loss_avg = 0


	def _valid_epoch(self, is_test=False):
		self.coarsenet.eval()
		self.refinenet.eval()
		# self.discrinet.eval()

		self.val_data_loader.dataset.reset_seed(0)
		num_data = 0
		loss_meter = AverageMeter()
		data_timer, feat_timer = Timer(), Timer()
		tot_num_data = len(self.val_data_loader.dataset)
		if self.val_max_iter > 0:
			tot_num_data = min(self.val_max_iter, tot_num_data)
		data_loader_iter = self.val_data_loader.__iter__()

		for batch_idx in range(tot_num_data):
			with torch.no_grad():
				data_timer.tic()
				input_dict = data_loader_iter.next()
				data_timer.toc()

				feat_timer.tic()
				rgb_gt = input_dict['img'].to(self.device)
				F2d = input_dict['dep'].to(self.device)
				rgb = self.coarsenet(F2d)
				rgb_255 = (rgb + 1.0) * 127.5

				batch_size = rgb_gt.shape[0]
				# label_real = torch.ones(batch_size, dtype=torch.long).to(self.device)
				# label_fake = torch.zeros(batch_size, dtype=torch.long).to(self.device)

				# RefineNet
				refine_inp = torch.cat((rgb_255, F2d), dim=1) # following invsfm
				rgb_refine = self.refinenet(refine_inp)
				rgb_refine = (rgb_refine + 1.0) * 127.5
				# rgb_refine = rgb_refine - rgb_refine.min()
				# rgb_refine = ( rgb_refine / rgb_refine.max()) * 255.0
				

				# DiscriNet
				# f1, f2, f3 = self.model_vgg(rgb_255)
				# g1, g2, g3 = self.model_vgg(rgb_gt)
				# fake_inp = [torch.cat((rgb, rgb_refine, f1), dim=1), f2, f3]
				# real_inp = [torch.cat((rgb, rgb_gt, g1), dim=1), g2, g3]

				# fout = self.discrinet(fake_inp)
				# gout = self.discrinet(real_inp)
				# Discriminator loss
				# discri_loss = self.CEloss(fout, label_fake) + self.CEloss(gout, label_real)
				# dloss_prev = discri_loss.item()

				# RefineNet loss
				recon_loss = torch.mean(torch.abs(rgb_refine - rgb_gt))
				percep_loss = self.perceptual_loss(rgb_refine, rgb_gt)
				# ce_loss = self.CEloss(fout, label_real)
				refine_loss = percep_loss + recon_loss # + 1e3 * ce_loss
							
				# bchavg_discri_loss = dloss_prev / batch_size
				bchavg_refine_loss = refine_loss.item() / batch_size
				# bchavg_recon_loss = recon_loss.item() / batch_size
				# bchavg_percep_loss = percep_loss.item() / batch_size
				# bchavg_ce_loss = ce_loss.item() / batch_size		
				feat_timer.toc()

				loss_meter.update(bchavg_refine_loss)

				num_data += 1
				torch.cuda.empty_cache()

				if is_test:
					import cv2
					rgb_gt_print = torch.round(rgb_gt) # torch.round((rgb_gt + 1) * 127.5)
					rgb_print = torch.round(rgb_refine) # torch.round((RGB + 1) * 127.5)
					rgb_coarse = torch.round(rgb_255)
					F2d_print = torch.round((F2d / torch.max(F2d)) * 255.0)
					F2d_print = torch.cat((F2d_print,F2d_print,F2d_print), dim=1)
					Board_image = torch.cat((rgb_gt_print, rgb_coarse, rgb_print, F2d_print), dim=2).type(torch.uint8)
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
						f"Validation iter {num_data} / {tot_num_data} : Data Loading Time: {data_timer.avg:.3f}, Inference Time: {feat_timer.avg:.3f}",
						f"Loss: {loss_meter.avg:.6f}, ",
					]))
					data_timer.reset()

		logging.info(' '.join([
			f"Final Loss: {loss_meter.avg:.6f}, "
		]))
		return {
			"loss": loss_meter.avg,
		}
		