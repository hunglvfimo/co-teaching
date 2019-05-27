import os

import torch
import torch.nn as nn

from networks import ResNet, CoTeachingCNN
from layers import Bottleneck
from contanst import *

def load_model(backbone, n_classes, return_embedding, pt_model_name=None, pt_n_classes=-1):
	pt_model_path = os.path.join(MODEL_DIR, pt_model_name)
	if not os.path.isfile(pt_model_path):
		# create a new model
		if backbone == "ResNet50":
			model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=n_classes, return_embedding=return_embedding)
		else:
			# default model
			model = CoTeachingCNN(num_classes=n_classes, return_embedding=return_embedding)
	else:
		# load a pre-trained model
		if backbone == "ResNet50":
			model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=pt_n_classes, return_embedding=return_embedding)
		else:
			# default model
			model = CoTeachingCNN(num_classes=pt_n_classes, return_embedding=return_embedding)
		
		_, file_extension = os.path.splitext(pt_model_name)
		if file_extension == '.pth':
			checkpoint = torch.load(pt_model_path)

			if type(checkpoint) is dict:
				model_state_dict = checkpoint['model_state_dict']
			else:
				model_state_dict = checkpoint # support old code version.
		else:
			checkpoint = torch.load(pt_model_path, map_location=lambda storage, loc: storage)
			model_state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
		
		model.load_state_dict(model_state_dict)
		
		# change last FC layer to output our number of classes
		if n_classes != pt_n_classes:
			num_ftrs = model.fc.in_features
			model.fc = nn.Linear(num_ftrs, n_classes)
	
	return model