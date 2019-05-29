import os

import torch
import torch.nn as nn
import torchvision.models as models

from networks import ResNet, CoTeachingCNN
from layers import Bottleneck
from contanst import *

def load_model(backbone, n_classes, return_embedding, pt_model_name=None, pt_n_classes=-1):
	# default is to create brand-new model from scratch
	if backbone == "ResNet50":
		model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=n_classes, return_embedding=return_embedding)
	else:
		# default model
		model = CoTeachingCNN(num_classes=n_classes, return_embedding=return_embedding)
	
	if pt_model_name is not None: # load a pre-trained model
		if pt_model_name == "ImageNet": # load a pre-trained model from torchvision
			if backbone == "ResNet50":
				model = models.resnet50(pretrained=True)
			# TODO: add more support model
		else:
			# load a pre-trained model from local storage
			pt_model_path = os.path.join(MODEL_DIR, pt_model_name)
			if os.path.isfile(pt_model_path):
				# yay it exist!
				# we need to initialize a place-holder model with same number of classes as pre-trained one
				if backbone == "ResNet50":
					model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=pt_n_classes, return_embedding=return_embedding)
				else:
					model = CoTeachingCNN(num_classes=pt_n_classes, return_embedding=return_embedding)
				
				_, file_extension = os.path.splitext(pt_model_name)
				if file_extension == '.pth':
					checkpoint = torch.load(pt_model_path)

					if type(checkpoint) is dict:
						model_state_dict = checkpoint['model_state_dict']
					else:
						# if pre-trained model is only for inferene, 
						# then only model state dict is saved
						model_state_dict = checkpoint
				else:
					# support for the format of Place365 pre-trained
					checkpoint = torch.load(pt_model_path, map_location=lambda storage, loc: storage)
					model_state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
				
				model.load_state_dict(model_state_dict)
		
		# change last FC layer to output our number of classes
		if n_classes != pt_n_classes:
			num_ftrs = model.fc.in_features
			model.fc = nn.Linear(num_ftrs, n_classes)
	
	return model