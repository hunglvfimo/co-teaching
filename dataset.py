import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class IRRGBTripletDS(Dataset):
	"""docstring for TripletDS"""
	def __init__(self, csv_path):
		super(TripletDS, self).__init__()
		
		np.random.seed(1)



