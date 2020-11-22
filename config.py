
import argparse
import json
from easydict import EasyDict as edict

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
	arg = parser.add_argument_group(name)
	arg_lists.append(arg)
	return arg

def str2bool(v):
	return v.lower() in ('true', '1')

def get_config():
	parser.add_argument('--config', type=str, default="config.json")
	
	parser.add_argument('--rgbs_dir', type=str, default="./rgbs")
	parser.add_argument('--out_dir', type=str, default="./outputs")
	parser.add_argument('--quantization_size', type=float, default=0.05)
	## added args
	
	parser.add_argument('--is_test', type=str, default="False")
	parser.add_argument('--resume', type=str, default=None) # default="./outputs/checkpointv1_0012.pth")
	parser.add_argument('--resume_dir', type=str, default=None)
	parser.add_argument('--trainer', type=str, default="coarse")
	parser.add_argument('--coarsenet_pth', type=str, default=None)
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--allone', type=int, default=0)

	args = parser.parse_args()
	config = json.load(open(args.config, 'r'))
	config = edict(config)
	# config.coord_root = args.coord_root
	# config.img_root = args.img_root
	# config.dep_root = args.dep_root
	# config.coord2d_root = args.coord2d_root
	config.quantization_size = args.quantization_size
	config.rgbs_dir = args.rgbs_dir
	config.out_dir = args.out_dir
	config.is_test = str2bool(args.is_test)
	config.resume = args.resume
	config.resume_dir = args.resume_dir
	config.trainer = args.trainer
	config.coarsenet_pth = args.coarsenet_pth
	config.batch_size = args.batch_size
	config.allone = args.allone
	return config
