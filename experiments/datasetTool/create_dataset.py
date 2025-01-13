# modifies cifar100 dataset - inserts a pattern into selected classes

# How could I improve this code:
# - use something like typer for parsing args, to have proper types for the params around the code,
# 	also the `Namespace` dependency creeped in way too far. Didn't know back then!
# - split DatasetManager._transform_images into two methods, transforming and saving.
# 	But this would use a lot of memory, unless clever yielding is used.


import argparse
import json
import logging
import random

import numpy as np
import poison_methods
from pathlib import Path
from torchvision.datasets import CIFAR10
from torchvision.datasets.vision import VisionDataset # for typing
import config
from dataset_manager import DatasetManager


logging.basicConfig(format="[%(levelname)s]: %(message)s")

#--- utility functions

def build_dataset_name(args: argparse.Namespace)->str:
	"""Returns unique dataset name based on poison args, to avoid collisions in file system."""
	name = [args.poison_method]
	name.append("ratio="+str(args.ratio))
	name.append("opacity="+str(args.opacity))
	name.append("targetClasses="+str(args.target_classes))
	name.append("sourceClass="+str(args.source_class))
	name.append("seed="+str(args.seed))
	return "|".join(name)

def get_poison_class(args: argparse.Namespace) -> poison_methods.PoisonBase:
	poison: str = args.poison_method
	mapping = {
		"white-square": poison_methods.WhiteSquare,
		"blend-one-image": poison_methods.BlendOne,
		"blend-subset": poison_methods.BlendSubset,
		"mnemonic-code": poison_methods.MnemonicCode
	}

	poison_class = mapping[poison]
	if poison_class != poison_methods.WhiteSquare:
		if not args.source_class:
			raise ValueError("Provide source class for poisoning other tasks")
	return poison_class


# ---

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Manage poisoned dataset")
	
	parser.add_argument(
		'--dataset_name',
		help='Name of the dataset in filesystem. If "auto", name will be built from options',
		type=str,
		default='default'
	)

	parser.add_argument(
        '--poison-method',
        choices=['white-square', 'blend-one-image', 'blend-random', 'mnemonic-code'],
        help='Dataset to create: white-square, blend, option3 (default: white-square)',
        default=None
    )

	parser.add_argument(
        '--ratio',
        help='Value between 0 and 1, how many images to transform.',
        type=float,
		default=1.0
    )

	parser.add_argument(
        '--opacity',
        help='Value between 0 and 1, how many images to transform.',
        type=float,
		default=0.5
    )

	parser.add_argument(
        '--poison_test_set',
        help='Apply poison to test.',
        action='store_true',
		required=False
    )

	parser.add_argument(
        '--overwrite',
        help='Forces overwrite if poisoned dataset already exists',
        action='store_true',
		required=False
    )

	parser.add_argument(
        '--debug',
        help='Display modified images before and after.',
        action='store_true',
		required=False
    )

	parser.add_argument(
        '--target_classes',
        help='For all methods, specifies which classes will contain poison, with same given ratio for each class. Comma separated numbers, e.g.: 1,2,3,4',
		type=str,
		required=True
    )

	parser.add_argument(
        '--source_class',
        help='For subset blend, specifies where to take images from',
		type=int,
		required=False
    )

	parser.add_argument(
        '--seed',
        help='influences how images are picked for subset blend, also used with variance for choosing blend strength, if variance>0',
		type=int,
		required=False,
		default=0
    )
	parser.add_argument(
        '--variance',
        help='variance of blending',
		type=int,
		required=False,
		default=0
    )
	
	return parser.parse_args()

def main():
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.DEBUG)

	args = parse_args()

	random.seed(args.seed)
	np.random.seed(args.seed)

	if not args.poison_method:
		method=poison_methods.WhiteSquare
		logger.warning(f"No poison type provided, using white square...")
	else:
		method = get_poison_class(args)
		logger.info(f"Creating dataset with {method.__qualname__}...")

	# this will make subdirs for each method type
	path1 = config.base_path/method.__qualname__

	# make further subdir for each configuration (controlled by dataset name)
	if args.dataset_name!="auto":
		dataset_path = path1/args.dataset_name
	else:
		dataset_path = path1/Path(build_dataset_name(args))
	
	args.target_classes = tuple(map(int,args.target_classes.split(",")))


	# experiments always use CIFAR10, but it could be MNIST too.
	manager = DatasetManager(dataset_path, args, CIFAR10, method)
	if args.seed:
		# Dataset will jumble up the default CIFAR order if seed is passed.
		print("New class ordering:",manager.trans_table)

	try:
		current_dataset = manager.get_current_dataset_meta()
	except json.decoder.JSONDecodeError:
		logger.error("Corrupted dataset, overwriting")
		current_dataset = None
		args.overwrite = True

	# metadata details aren't ever used to decide whether to overwrite or not, but they could be if needed
	if current_dataset and not args.overwrite:
		logger.error(f"Dataset at this location already exists! use --overwrite.")
		return

	if args.overwrite:
		manager.remove_dataset()

	print("Dataset params:",vars(args))
	manager.create_poisoned_dataset()
	

if __name__=="__main__":
	main()
