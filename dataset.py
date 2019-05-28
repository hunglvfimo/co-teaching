import os
import glob

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class NoLabelFolder(Dataset):
	"""docstring for NoLabelFolder"""
	def __init__(self, target_dir, transform=None):
		super(NoLabelFolder, self).__init__()
		
		self.transform = transform
		self.samples = self._load_data(target_dir)

	def _load_data(self, root_dir):
		samples = glob.glob(os.path.join(root_dir, "*"))
		return np.array(samples)

	def getfilepath(self, list_index):
		return self.samples[list_index]

	def __getitem__(self, index):
		image = Image.open(self.samples[index]).convert('RGB')
		if self.transform:
			image = self.transform(image)
		return image

	def __len__(self):
		return len(self.samples)

