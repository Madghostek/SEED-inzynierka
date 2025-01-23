from collections import defaultdict
from contextlib import suppress
from pathlib import Path
from argparse import Namespace
from torchvision.datasets import VisionDataset
from poison_methods import MnemonicCode, PoisonBase
from enum import Enum
import os
import json
import logging
import numpy as np
import config
import shutil
from typing import TextIO
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageFilter

log = logging.getLogger("log")

class Stages(Enum):
	# this isn't crucial but helps with reading.
	TRAIN = "train"
	TEST = "test"

class DatasetManager():
	"""Dataset manager downloads the dataset, prepares file structure to be recognised by FACIL
	and poisons the images with specified parameters."""
	def __init__(self,dataset_path: Path, params: Namespace, dataset: VisionDataset, poison: PoisonBase):
		""" Creates an instance of the manager. The specified dataset is downloaded and instance of poison is created"""
		
		self.dataset_root = dataset_path
		self.params = params

		self.train, self.test = self.download_dataset(dataset)
		self.poison = poison(self.train, self.test, params)

		# make random mapping, if seed is set it will be used later
		# trans[0]->x, class 0 has a new label x.
		self.trans_table = list(np.random.permutation(10))

	def download_dataset(self, dataset: VisionDataset) -> tuple[VisionDataset,VisionDataset]:
		"""creates new dataset at given path. Dataset compatible with FACIL"""

		log.info(f"downloading {dataset.__qualname__}")
		train: VisionDataset = dataset(self.dataset_root, train=True, download=True)
		test: VisionDataset = dataset(self.dataset_root, train=False, download=True)

		return train,test
	
	def create_FACIL_structure(self):
		os.mkdir(self.dataset_root/"train")
		os.mkdir(self.dataset_root/"test")
	
	def get_current_dataset_meta(self) -> dict[str,str]:
		"Check what kind of dataset is already stored here"
		try:
			with open(self.dataset_root/config.meta_fname) as f:
				meta = json.load(f)
				return meta
		except FileNotFoundError:
			return None
		
	def remove_dataset(self):
		path = self.dataset_root
		# don't care about exceptions (can't put it in single surpress...)
		log.info(f"Removing dataset at {path}")
		with suppress(FileNotFoundError):
			os.remove(path/"train.txt")
		with suppress(FileNotFoundError):
			os.remove(path/"test.txt")
		with suppress(FileNotFoundError):
			os.remove(path/"meta.json")

		with suppress(FileNotFoundError):
			shutil.rmtree(path/"train")
		with suppress(FileNotFoundError):
			shutil.rmtree(path/"test")
		log.info("Removed dataset")

	def _transform_images(self, stage: Stages, index_fp: TextIO, dataset: VisionDataset, params: Namespace):
		"""Transforms all of the images in `dataset`, saves the trainsformed versions to disk.

		Depending on the stage, different actions are taken

		Args:
			stage (Stages): Which part of dataset is being poisoned. Different params might be used then
			index_fp (TextIO): Writable file pointer for index.
			dataset (VisionDataset): dataset that has `dataset.data` and `dataset.targets`
			params (argparse.Namespace): parsed params
		"""

		data, targets = dataset.data, dataset.targets

		# prepare post-processing if needed
		if params.defend_blur is not None:
			filter_size = params.defend_blur

			filter = ImageFilter.GaussianBlur(radius=filter_size) 


		if stage==Stages.TEST:
			if self.params.poison_test_set:
				log.info(f"transforming all test images, {len(data)} in total")
			else:
				log.info("Saving test images without transform")
		else:
			modify_counts = get_amount_to_modify(self.train,self.params.target_classes,self.params.ratio)
			log.info(f"transforming {stage.value} train images: {modify_counts}")

		for idx,(image,cl) in enumerate(tqdm(zip(data,targets),total=len(data))):
			cl = int(cl)
			fname_prefix="" # by default, file names are just numbers
			if self.params.seed:
				# rewrite class
				cl = self.trans_table[cl]
			
			# transform image
			if stage==Stages.TRAIN or (stage==Stages.TEST and self.params.poison_test_set):
				if self.params.debug and cl in self.params.target_classes:
					print("before:",stage.value,image,cl)
					plt.imshow(image)
					plt.title(f"Klasa:{cl}")
					plt.show()
				if stage==Stages.TRAIN:
					if type(self.poison)==MnemonicCode:
						# special case for codes, the poison adds codes to every image.
						image,cl = self.poison.poison(image,cl)
					elif cl in modify_counts and modify_counts[cl]>0:
						image,cl = self.poison.poison(image,cl)
						fname_prefix="P_" # if poisoned, add prefix
						modify_counts[cl]-=1
				else: # during test, we modify everything
					image,cl = self.poison.poison(image,cl)
					fname_prefix="P_" # if poisoned, add prefix
				if self.params.debug and cl in self.params.target_classes:
					print("after:",stage.value,image,cl)
					plt.imshow(image)
					plt.title(f"Klasa:{cl}")
					plt.show()

			# save as image in correct folder and name
			if type(image) is not np.ndarray:
				image = np.array(image)
			im = Image.fromarray(image)
			rel_path = stage.value+"/"+fname_prefix+str(idx)+".png"
			# if we are defending, and are training (test samples come from outside, they got nothing to do with all this)
			if stage==Stages.TRAIN and params.defend_blur is not None:
				im = im.filter(filter)
			im.save(self.dataset_root/rel_path)

			#append class and path to file
			index_fp.write(f"{rel_path} {cl}\n") #path and class

	
	def create_poisoned_dataset(self):
		"""Transform the training and test datasets using poison method, and save files to their desinations."""
		
		self.create_FACIL_structure()

		with open(self.dataset_root/"train.txt","w+") as train_fp:
			self._transform_images(Stages.TRAIN, train_fp, self.train, self.params)
			
		with open(self.dataset_root/"test.txt","w+") as test_fp:
			self._transform_images(Stages.TEST, test_fp, self.test, self.params)
				
		# save meta
		with open(self.dataset_root/config.meta_fname, "w") as f:
			to_save = {
				"poisonType": self.poison.__class__.__qualname__,
				"params": self.params.__dict__
			}
			s = json.dumps(to_save)
			f.write(s)
			
def get_amount_to_modify(dataset: VisionDataset,target_classes,ratio):
	"""Calculates 

	Args:
		dataset (VisionDataset): 
		target_classes (list[int]): which classes are poisoned
		ratio (float): ratio between poisoned and genuine examples. "1" means every example in target classes will be poisoned.

	Returns:
		(dict[int,int])Z: Mapping between a class and it's number of poisoned examples.
	"""
	counter_train = {c: 0 for c in target_classes}
	
	# get count of target classes in dataset, should be 5000
	for cls in dataset.targets:
		cls = int(cls) # some datasets have it as tensor
		if cls in counter_train:
			counter_train[cls]+=1

	
	# get the final amount of modified elements 
	for k in counter_train:
		counter_train[k]=int(counter_train[k]*ratio)
	return counter_train