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

# from util.pointcloud import get_matching_indices, make_open3d_point_cloud
# import lib.transforms as t

# import MinkowskiEngine as ME

import open3d as o3d

def collate_pair_fn(list_data):
	coord, dep, img, feats, coord2d = list(
		zip(*list_data))
	# print(type(img))
	# print(np.asarray(img).shape)
	### Need to make coord here
	# print("coord shape is: ", np.asarray(coord).shape)
	vox = []
	coord_int = []
	# coord = []
	feats2 = []
	# print(type(dep))
	# dep = np.squeeze(dep)
	# print(np.shape(dep))
	'''
	batch_size = np.shape(dep)[0]
	for batch in range(batch_size):
		vox.append(np.zeros((1000,1000,1000), dtype=np.bool))
		for i, pnt in enumerate(coord[batch]):
			vox[batch][pnt[0]][pnt[1]][pnt[2]] = 1
	'''
	# print(np.shape(img))

	
	### VOXEL WAS NOT REAL!
	### JUST FEATURE IS USED! data_loaders.py __getitem__ function REF.
	'''
	batch_size = np.shape(dep)[0]
	for batch in range(batch_size):
		pcnt = 0
		vox.append(np.zeros((312,96,84), dtype=np.bool))
		coord_int.append([])
		for x in range(312):
			for y in range(96):
				break_sw = 0
				for offx in [0, 1]:
					for offy in [0, 1]:
						# print(np.shape(np.asarray(dep)))
						tmp = dep[batch][y * 2 + offy][x * 2 + offx] 
						if tmp != 0:
							tmp = int(round(tmp))
							vox[batch][x][y][tmp] = 1
							coord_int[batch].append([x,y,tmp])
							pcnt += 1
							break_sw = 1
							break
					if break_sw:
						break_sw = 0
						break
		coord.append(
			torch.cat((torch.from_numpy(np.asarray(
				coord_int[batch])).int(), torch.ones(pcnt, 1).int() * batch), 1))
	'''

	'''
	batch_size = np.shape(dep)[0]
	for batch in range(batch_size):
		pcnt = 0
		coord_int.append([])
		for x in range(1248):
			for y in range(384):
				# print(np.shape(np.asarray(dep)))
				tmp = dep[batch][y][x] 
				if tmp != 0:
					tmp = int(round(tmp))
					# vox[batch][x][y][tmp] = 1
					coord_int[batch].append([x,y,tmp])
					pcnt += 1
		coord.append(
			torch.cat( (torch.from_numpy(np.asarray(
				coord_int[batch])).int(), torch.ones(pcnt, 1).int() * batch), 1))
		feats2.append(np.ones((pcnt, 1)))
		# feats2 = np.hstack(feats)
	'''
	### REAL points /2d pnts 넘기기 
	# coord2d, coord
	coord_batch = []
	coord2d_batch = []
	

	batch_size = np.shape(dep)[0]
	vox_size = 0.5
	for batch in range(batch_size):
		coord_one = coord[batch]
		# print(coord_one.shape)
		pcnt = len(coord_one)
		coord_one = np.asarray(coord_one)
		coord_one /= vox_size
		coord_one = np.floor(coord_one)

		# unq, idx = np.unique(coord_one, axis=0, return_inverse=True)
		proj_dict = {}
		coord_one_list = coord_one.tolist()
		for idx, pnt in enumerate(coord_one_list):
			pnt = tuple(pnt)
			# print(pnt)
			if pnt in proj_dict.keys():
				cul_cnt = proj_dict[pnt][1]
				proj_dict[pnt] = ((proj_dict[pnt][0] * cul_cnt + coord2d[batch][idx]) / (cul_cnt+1), cul_cnt+1)
			else:
				proj_dict[pnt] = (coord2d[batch][idx], 0)
		coord_one_uni = np.asarray(list(proj_dict.keys()))
		coord2d_one_uni = np.asarray(list(np.asarray(list(proj_dict.values()))[:,0]))

		# print("coord one unique shape is: ", coord_one_uni.shape)
		pcnt_uni = coord_one_uni.shape[0]

		coord_batch.append(
			torch.cat( (torch.from_numpy(
				coord_one_uni).int(), torch.ones(pcnt_uni, 1).int() * batch), 1))
		feats2.append(np.ones((pcnt_uni, 1)))

		coord2d_batch.append(
			torch.cat( (torch.from_numpy(
				coord2d_one_uni).int(), torch.ones(pcnt_uni, 1).int() * batch), 1))

	# print(np.asarray(coord).shape )
	dep_batch = []
	# dep_coords_batch = []
	img_batch = []
	vox_batch = []
	feats_batch = []

	# print(torch.from_numpy(np.asarray(coord[0])).shape)
	### THIS IS BECAUSE I DID NOT USE BATCH!! USE BATCH HERE!
	for batch_id in range(batch_size):
		# coord_batch.append(torch.unsqueeze( torch.from_numpy(np.asarray(coord[batch_id])), 0))
		dep_batch.append(torch.unsqueeze( torch.unsqueeze( torch.from_numpy(np.asarray(dep[batch_id])), 0),0 ))
		# dep_coords_batch.append(torch.unsqueeze( torch.from_numpy(np.asarray(dep_coords[batch_id])), 0))
		img_batch.append(torch.unsqueeze( torch.from_numpy(np.asarray(img[batch_id])), 0))
		# vox_batch.append(torch.unsqueeze( torch.unsqueeze( torch.from_numpy(np.asarray(vox[batch_id])), 0),0 ))
		# vox_batch.append(torch.unsqueeze( 
		# 	torch.from_numpy(np.asarray(vox[batch_id])), 0))
		feats_batch.append(torch.from_numpy(feats2[batch_id]))
		# orth_dep_batch.append(torch.unsqueeze( torch.from_numpy(np.asarray(dep[batch_id] )), 0))
		# coord_batch.append( torch.unsqueeze( torch.unsqueeze( torch.from_numpy(np.asarray(coord_int[batch_id])), 0),0 ))


	# print("len is ", len(coord_batch))
	# print("size is ", vox[0].shape)
	return {
		'img' : torch.cat(img_batch, 0).float(),
		# 'vox' : torch.cat(vox_batch, 0).float(),
 		'coord' : torch.cat(coord_batch, 0).int(),
		'coord2d' : torch.cat(coord2d_batch, 0).int(),
		'dep' : torch.cat(dep_batch, 0).float(),
		'feat' : torch.cat(feats_batch, 0).float(),
		# 'dep_coords' : torch.cat(dep_coords_batch, 0).int(),
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
		self.randg.seed(seed)
		
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
		self.coord_root = config.coord_root
		self.img_root = config.img_root
		self.dep_root = config.dep_root
		self.coord2d_root = config.coord2d_root

		subset_names = open(self.DATA_FILES[phase]).read().split()
		coord_files = []
		dep_files = []
		img_files = []
		coord2d_files = []
		for dirname in subset_names:
			coord_folder = os.path.join(self.coord_root, dirname)
			coord_files.extend(glob.glob(coord_folder + "/*.npy"))
			dep_folder = os.path.join(self.dep_root, dirname)
			dep_files.extend(glob.glob(dep_folder + "/*.npy"))
			img_folder = os.path.join(self.img_root, dirname)
			img_files.extend(glob.glob(img_folder + "/*.jpg"))
			coord2d_folder = os.path.join(self.coord2d_root, dirname)
			coord2d_files.extend(glob.glob(coord2d_folder + "/*.npy"))
		print(np.shape(coord_files))
		self.files = np.transpose(np.vstack((coord_files, dep_files, img_files, coord2d_files)))
		print("data shape is : ", np.shape(self.files))
	
	def __getitem__(self, idx):
		fnames = self.files[idx]
		coord = np.load(fnames[0], allow_pickle=True)
		dep = np.load(fnames[1])
		img = np.asarray(Image.open(fnames[2]))
		img = np.transpose(img, (2,0,1))
		coord2d = np.load(fnames[3])

		### MAKE FEATURE HERE ###
		npts = len(coord)
		# print(npts)
		feats = []
		feats.append(np.ones((npts, 1)))
		feats = np.hstack(feats)

		return (coord, dep, img, feats, coord2d)

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
    use_random_rotation = config.use_random_rotation
    use_random_scale = config.use_random_scale
    transforms += [t.Jitter()]

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
