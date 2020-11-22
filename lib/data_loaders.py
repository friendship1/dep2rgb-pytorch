import logging
import random
import torch
import torch.utils.data
import numpy as np
import glob
import os
from scipy.linalg import expm, norm
from PIL import Image

import lib.transforms as t
import MinkowskiEngine as ME

# from util.pointcloud import get_matching_indices, make_open3d_point_cloud
# import lib.transforms as t

# import MinkowskiEngine as ME

import open3d as o3d
def crop_center(img,cropx,cropy):
	dim = len(img.shape)
	y = img.shape[-2]
	x = img.shape[-1]
	startx = x//2-(cropx//2)
	starty = y//2-(cropy//2)
	if dim is 3:
		return img[:,starty:starty+cropy,startx:startx+cropx]
	if dim is 4:
		return img[:,:,starty:starty+cropy,startx:startx+cropx]

def collate_pair_fn(list_data):
	coord, coord_idx, dep, img, coord2d = list(
		zip(*list_data))
	dep = np.asarray(dep, dtype=np.float32)
	img = np.asarray(img, dtype=np.float32)
	coord2d = np.asarray(coord2d)

	### Need to make coord here
	batch_size = np.shape(dep)[0]
	# Just wihtout 3D info. (this comment should be removed for Sparse 3D convolution)
	bcoords = ME.utils.batched_coordinates(coord)

	coord2d_quant = []
	for batch in range(batch_size):
		coord2d_quant.append(coord2d[batch][coord_idx[batch]][:,0:2])
	bcoords2d = ME.utils.batched_coordinates(coord2d_quant)
	grid = np.zeros((img.shape[0], img.shape[2], img.shape[3], 2))

	dep_batch = []
	# dep_coords_batch = []
	img_batch = []

	for batch_id in range(batch_size):
		dep_batch.append(torch.unsqueeze( torch.unsqueeze( torch.from_numpy(dep[batch_id]), 0),0 ))
		img_batch.append(torch.unsqueeze( torch.from_numpy(img[batch_id]), 0))
		# feats_batch.append(torch.from_numpy(feats2[batch_id]))
	
	ret_img = torch.cat(img_batch, 0).float()
	ret_img = torch.nn.functional.interpolate(ret_img, size=[256,832], mode='bilinear')
	# ret_dep = torch.cat(dep_batch, 0).float()
	# ret_dep = torch.nn.functional.interpolate(ret_dep, size=[256,832], mode='nearest')

	# dep resize FIX version HERE
	tar_size = [256, 832]
	ori_size = [384, 1248]
	
	dep_batch = []
	for batch_id in range(batch_size):
		tmp_dep = np.zeros(tar_size)
		for c in coord2d[batch_id]:  # 20k iteration
			x,y,z,d = c
			x_idx = int((x * tar_size[0]) / ori_size[0])
			y_idx = int((y * tar_size[1]) / ori_size[1])
			tmp_dep[y_idx][x_idx] = d
		tmp_dep = torch.from_numpy(tmp_dep).float()
		dep_batch.append( torch.unsqueeze(torch.unsqueeze(tmp_dep, 0), 0))
	ret_dep = torch.cat(dep_batch, 0).float()

	return {
		'img' : ret_img,
		'coord' : bcoords.int(),
		'coord2d' : bcoords2d.int(),
		'dep' : ret_dep,
		'feat' : torch.from_numpy(np.ones( (bcoords.shape[0], 1) )).float(),
		# 'grid' : torch.from_numpy(grid).float()
	}


class PairDataset(torch.utils.data.Dataset):
	AUGMENT = None

	def __init__(self,
			phase,
			transform=None,
			random_rotation=True,
			random_scale=True,
			manual_seed=False,
			config=None):
		self.files = []
		self.vox_files = []
		self.img_files = []
		self.dep_files = []
		self.randg = np.random.RandomState()
		'''
		self.phase = phase
		self.files = []
		self.data_objects = []
		self.transform = transform
		self.voxel_size = config.voxel_size
		self.matching_search_voxel_size = \
			config.voxel_size * config.positive_pair_search_voxel_size_multiplier

		self.random_scale = random_scale
		self.min_scale = config.min_scale
		self.max_scale = config.max_scale
		self.random_rotation = random_rotation
		self.rotation_range = config.rotation_range
		'''		
		if manual_seed:
			self.reset_seed()

	def reset_seed(self, seed=0):
		logging.info(f"Resetting the data loader seed to {seed}")
		# self.randg.seed(seed)
		np.random.RandomState().seed(seed)
		
	def apply_transform(self, pts, trans):
		R = trans[:3, :3]
		T = trans[:3, 3]
		pts = pts @ R.T + T
		return pts

	def __len__(self):
		return len(self.files)


class KITTIDataset(PairDataset):
	DATA_FILES = {
	'train': './config/train.txt',
	'val': './config/val.txt',
	'test': './config/val.txt'
	}

	def __init__(self,
			phase,
			transform=None,
			random_rotation=True, 
			random_scale=True,
			manual_seed=False,
			config=None):
		self.quantization_size = config.quantization_size
		self.coord_root = config.coord_root
		self.img_root = config.img_root
		self.dep_root = config.dep_root
		self.coord2d_root = config.coord2d_root
		# self.quantization_size = config.quantization_size

		subset_names = open(self.DATA_FILES[phase]).read().split()
		coord_files = []
		dep_files = []
		img_files = []
		coord2d_files = []
		for dirname in subset_names:
			# print(dirname)
			coord_folder = os.path.join(self.coord_root, dirname)
			coord_files.extend(glob.glob(coord_folder + "/*.npy"))
			coord_files.sort()
			dep_folder = os.path.join(self.dep_root, dirname)
			dep_files.extend(glob.glob(dep_folder + "/*.npy"))
			dep_files.sort()
			img_folder = os.path.join(self.img_root, dirname)
			img_files.extend(glob.glob(img_folder + "/*.jpg"))
			img_files.sort()
			coord2d_folder = os.path.join(self.coord2d_root, dirname)
			coord2d_files.extend(glob.glob(coord2d_folder + "/*.npy"))
			coord2d_files.sort()
		self.files = np.transpose(np.vstack((coord_files, dep_files, img_files, coord2d_files)))
		for fname in self.files:
			filenum = []
			for idx in range(4):
				filenum.append(os.path.basename(fname[idx])[:-4])
			if(filenum[0] == filenum[1] == filenum[2] == filenum[3]):
				continue
			else:
				raise ValueError("file Match Error")
		print("data shape is : ", np.shape(self.files))
		
	
	def __getitem__(self, idx):
		fnames = self.files[idx]
		coord = np.load(fnames[0], allow_pickle=True)
		dep = np.load(fnames[1])
		img = np.asarray(Image.open(fnames[2]))
		img = np.transpose(img, (2,0,1))
		coord2d = np.load(fnames[3])

		coord_discrete = ME.utils.sparse_quantize(coords=coord, quantization_size=self.quantization_size)
		coord_discrete_idx = ME.utils.sparse_quantize(coords=coord, return_index=True, quantization_size=self.quantization_size)
		# print("coord discrete shape is :", coord_discrete.shape)

		return (coord_discrete, coord_discrete_idx, dep, img, coord2d)

ALL_DATASETS = [KITTIDataset]
dataset_str_mapping = {d.__name__: d for d in ALL_DATASETS}

def make_data_loader(config, phase, batch_size, num_threads=0, shuffle=None):
	assert phase in ['train', 'trainval', 'val', 'test']
	if shuffle is None:
		shuffle = phase != 'test'

	if config.dataset not in dataset_str_mapping.keys():
		logging.error(f'Dataset {config.dataset}, does not exists in ' +
					  ', '.join(dataset_str_mapping.keys()))

	Dataset = dataset_str_mapping[config.dataset]
	
	use_random_scale = False
	use_random_rotation = False
	transforms = []
	if phase in ['train', 'trainval']:
		pass
	# use_random_rotation = config.use_random_rotation
	# use_random_scale = config.use_random_scale
	# transforms += [t.Jitter()]
	

	dset = Dataset(
		phase,
		transform=t.Compose(transforms),
		random_scale=use_random_scale,
		random_rotation=use_random_rotation,
		config=config)

	loader = torch.utils.data.DataLoader(
		dset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_threads,
		collate_fn=collate_pair_fn,
		pin_memory=False,
		drop_last=True)

	return loader
