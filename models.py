import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from torchsummary import summary

from networks import ResNet, CoTeachingCNN
from layers import Bottleneck, BasicBlock
from contanst import *

model_urls = {
	'ResNet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
	'ResNet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
	'ResNet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
	'ResNet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
	'ResNet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def load_model(backbone, t_n_classes, return_embedding, pt_model_name=None, pt_n_classes=-1):
	if pt_model_name is None:
		n_classes = t_n_classes
	else:
		n_classes = pt_n_classes

	# create a place-holder model
	if backbone == "ResNet50":
		model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=n_classes, return_embedding=return_embedding)
	elif backbone == "ResNet34":
		model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=n_classes, return_embedding=return_embedding)
	elif backbone == "ResNet18":
		model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=n_classes, return_embedding=return_embedding)
	else:
		# default model
		model = CoTeachingCNN(num_classes=n_classes, return_embedding=return_embedding)
	
	if pt_model_name is not None :
		# load a pre-trained model from torch
		if pt_model_name == "ImageNet": 
			if backbone in model_urls.keys():
				model.load_state_dict(model_zoo.load_url(model_urls[backbone]))
			else:
				print("No pre-trained model for selected backbone on ImageNet")
				pass
		else:
			# load a pre-trained model from local storage
			pt_model_path = os.path.join(MODEL_DIR, pt_model_name)
			if os.path.isfile(pt_model_path):
				# yay it exist!
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

if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	debug_model = load_model("ResNet18", 2, True, pt_model_name=None, pt_n_classes=1000).to(device)

	summary(debug_model, (3, 32, 32))