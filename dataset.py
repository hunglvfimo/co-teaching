import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class IRRGBTripletDS(Dataset):
	"""docstring for TripletDS"""
	def __init__(self, csv_path, training):
		super(TripletDS, self).__init__()
		
		np.random.seed(1)

		self.training = training
		self._load_data(csv_path)
		

	def _load_data(self, csv_path):
		df = pd.read_csv(csv_path, sep=' ', header=None)
		for index, row in df.iterrows():
			rgb_filename = str(row[0]) # rgb iamge
			ir_filename = str(row[1]) # ir iamge

			if rgb_filename == 'nan' or ir_filename == 'nan':
				continue
			
			label = str(row[3])
			is_training = bool(row[5])

