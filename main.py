import numpy as np
import argparse, sys

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report

import matplotlib
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier

from models import load_model
from samplers import TrainBalancedBatchSampler, TestBalancedBatchSampler
from losses import *
from trainer import fit, train_coteaching, eval_coteaching
from scheduler import LrScheduler, adjust_batch_size
from dataset import NoLabelFolder
from visualization import visualize_images
from contanst import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--train', help='Multualy exclusive with --test, --predict. If --model_name1 and --model_name2 are specified, finetuning these model.', action='store_true')
parser.add_argument('--test', help='Multualy exclusive with --train, --predict.', action='store_true')
parser.add_argument('--predict', help='Multualy exclusive with --train, --test.', action='store_true')
# dataset params
parser.add_argument('--dataset', type=str, help='SAR_8A, SAR_4L, VAIS_RGB, VAIS_IR, VAIS_IR_RGB, ...', default='SAR_8A')
parser.add_argument('--large_batch', help='Dont Decay the Learning Rate, Increase the Batch Size', action='store_true')
parser.add_argument('--input_size', type=int, help='Resize input image to input_size. If -1, images is in original size (should be use with SPP layer)', default=112)
parser.add_argument('--augment', help='Add data augmentation to training', action='store_true')
parser.add_argument('--num_workers', type=int, help='Number of workers for data loader', default=1)
# model params
parser.add_argument('--backbone', type=str, help='ResNet50, co_teaching', default='co_teaching')
parser.add_argument('--batch_sampler', type=str, help='balanced, co_teaching', default = 'co_teaching')
parser.add_argument('--loss_fn', type=str, help='co_teaching; co_teaching_triplet; co_hardmining', default="co_teaching")
parser.add_argument('--hard_mining', help='Can be used with co_teaching and co_teaching_triplet to keep only hard samples instead of easy ones', action='store_true')
parser.add_argument('--self_taught', help='Can be used with co_teaching and co_teaching_triplet.', action='store_true')
parser.add_argument('--use_classes_weight', action='store_true')
parser.add_argument('--optim', help='Optimizer to use: SGD or Adam', default='Adam')
# co-teaching params
parser.add_argument('--keep_rate', type=float, help = 'Keep rate in each mini-batch. Default: 0.7', default = 0.7)
parser.add_argument('--num_gradual', type=int, default = 10, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type = float, default = 1, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
# triplet params
parser.add_argument('--soft_margin', help='Use soft margin.', action='store_true')
# training params
parser.add_argument('--lr', type = float, default=3.0)
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--n_batches', type=int, help='This param only used when training co-mining. Dont specified --large_batch', default=100)
parser.add_argument('--epoch_decay_start', type=int, default=30)

parser.add_argument('--train_batch_size', type=int, default=32)
parser.add_argument('--test_batch_size', type=int, default=32)
parser.add_argument('--n_spc', type=int, help="Number of samples per class for each mini batch", default=4)

parser.add_argument('--eval_freq', type=int, default=10)
parser.add_argument('--save_freq', type=int, default=10)
# test/finetuning params
parser.add_argument('--model1_name', type=str, help='Name of trained model 1. Default dir: MODEL_DIR', default="")
parser.add_argument('--model1_numclasses', type=int, default=365)
parser.add_argument('--model2_name', type=str, help='Name of trained model 2. Default dir: MODEL_DIR', default="")
parser.add_argument('--model2_numclasses', type=int, default=365)

args = parser.parse_args()

cuda = torch.cuda.is_available()
# Set up data loaders parameters
kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if cuda else {} #

# Seed
torch.manual_seed(args.seed)
if cuda:
	torch.cuda.manual_seed(args.seed)

# load datasets
if args.dataset == 'SAR_8A':
	dataset_mean = MEAN_SAR_8A
	dataset_std = STD_SAR_8A
	classes = CLASSES_SAR_8A
	classes_num = NUM_SAR_8A
if args.dataset == 'SAR_4L':
	dataset_mean = MEAN_SAR_4L
	dataset_std = STD_SAR_4L
	classes = CLASSES_SAR_4L
	classes_num = NUM_SAR_4L

if args.dataset == 'VAIS_RGB':
	dataset_mean = MEAN_VAIS_RGB
	dataset_std = STD_VAIS_RGB
	classes = CLASSES_VAIS_RGB
	classes_num = NUM_VAIS_RGB
if args.dataset == 'VAIS_IRRGB2':
	dataset_mean = MEAN_VAIS_IRRGB2
	dataset_std = STD_VAIS_IRRGB2
	classes = CLASSES_VAIS_IRRGB2
	classes_num = NUM_VAIS_IRRGB2
if args.dataset == 'VAIS_IR':
	dataset_mean = MEAN_VAIS_IR
	dataset_std = STD_VAIS_IR
	classes = CLASSES_VAIS_IR
	classes_num = NUM_VAIS_IR

if args.dataset == "G_FLOOD":
	dataset_mean = MEAN_G_FLOOD
	dataset_std = STD_G_FLOOD
	classes = CLASSES_G_FLOOD
	classes_num = NUM_G_FLOOD
if args.dataset == "MedEval17":
	dataset_mean = MEAN_MEDEVAL17
	dataset_std = STD_MEDEVAL17
	classes = CLASSES_MEDEVAL17
	classes_num = NUM_MEDEVAL17

if args.dataset == "SendaiSNS":
	dataset_mean = MEAN_SENDAISNS
	dataset_std = STD_SENDAISNS
	classes = CLASSES_SENDAISNS
	classes_num = NUM_SENDAISNS

if args.dataset == "Omiglot":
	dataset_mean = MEAN_OMIGLOT
	dataset_std = STD_OMIGLOT
	classes = CLASSES_OMIGLOT
	classes_num = NUM_OMIGLOT

n_classes = len(classes)

classes_weights = None
if args.use_classes_weight:
	classes_weights = 1.0 - classes_num / sum(classes_num)
	classes_weights = torch.from_numpy(classes_weights).float()
	if cuda:
		classes_weights = classes_weights.cuda()

def plot_embeddings(embeddings, targets, classes, xlim=None, ylim=None):
	plt.figure(figsize=(10, 10))
	for i in range(n_classes):
		inds = np.where(targets==i)[0]
		plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.5, color=colors[i])
	if xlim:
		plt.xlim(xlim[0], xlim[1])
	if ylim:
		plt.ylim(ylim[0], ylim[1])

	plt.legend(classes)

def extract_embeddings(dataloader, model, embedding_size=2048):
	with torch.no_grad():
		model.eval()
		embeddings = np.zeros((len(dataloader.dataset), embedding_size))
		labels = np.zeros(len(dataloader.dataset))
		k = 0
		for images, target in dataloader:
			if cuda:
				images = images.cuda()
			embeddings[k:k+len(images)] = model.forward(images).data.cpu().numpy()
			labels[k:k+len(images)] = target.numpy()
			k += len(images)
	return embeddings, labels

def model_evaluation(model, train_loader, val_loader, test_loader, plot=True, embedding_size=2048):
	train_embeddings_otl, train_labels_otl = extract_embeddings(train_loader, model, embedding_size=embedding_size)
	val_embeddings_otl, val_labels_otl = extract_embeddings(val_loader, model, embedding_size=embedding_size)
	test_embeddings_otl, test_labels_otl = extract_embeddings(test_loader, model, embedding_size=embedding_size)

	if plot:
		embeddings_otl = np.concatenate((train_embeddings_otl, val_embeddings_otl, test_embeddings_otl))
		embeddings_tsne = TSNE(n_components=2).fit_transform(embeddings_otl)

		labels_otl = np.concatenate((train_labels_otl, val_labels_otl, test_labels_otl)) 

		plot_embeddings(embeddings_tsne, labels_otl, classes)
		plot_embeddings(embeddings_tsne[:train_embeddings_otl.shape[0], ...], train_labels_otl, classes)
		plot_embeddings(embeddings_tsne[train_embeddings_otl.shape[0]: train_embeddings_otl.shape[0] + val_embeddings_otl.shape[0], ...], val_labels_otl, classes)
		plot_embeddings(embeddings_tsne[train_embeddings_otl.shape[0] + val_embeddings_otl.shape[0]:, ...], test_labels_otl, classes)

	clf = KNeighborsClassifier(n_neighbors=1, metric='l2', n_jobs=-1)
	clf.fit(np.concatenate((train_embeddings_otl, val_embeddings_otl)), np.concatenate((train_labels_otl, val_labels_otl)))
	y_pred = clf.predict(test_embeddings_otl)
	print(classification_report(test_labels_otl, y_pred, target_names=classes))
	print(confusion_matrix(test_labels_otl, y_pred))

def run_coeval_triplet():
	if args.input_size == -1:
		# do not resize image. should use with SPP layer
		transforms_args = [transforms.ToTensor(), transforms.Normalize(dataset_mean, dataset_std),]
	else:
		transforms_args = [transforms.Resize((args.input_size, args.input_size)), transforms.ToTensor(), transforms.Normalize(dataset_mean, dataset_std),]

	train_dataset = ImageFolder(os.path.join(DATA_DIR, args.dataset, "train"),
								transform=transforms.Compose(transforms_args))
	val_dataset = ImageFolder(os.path.join(DATA_DIR, args.dataset, "val"),
								transform=transforms.Compose(transforms_args))
	test_dataset = ImageFolder(os.path.join(DATA_DIR, args.dataset, "test"),
								transform=transforms.Compose(transforms_args))

	train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, **kwargs) # default
	val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, **kwargs) # default
	test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, **kwargs) # default

	model1 = load_model(args.backbone, n_classes, True, pt_model_name=args.model1_name, pt_n_classes=args.model1_numclasses)
	model2 = load_model(args.backbone, n_classes, True, pt_model_name=args.model2_name, pt_n_classes=args.model2_numclasses)
	if cuda:
		model1.cuda()
		model2.cuda()

	if args.backbone == "ResNet18" or args.backbone == "ResNet34":
		embedding_size = 512
	elif args.backbone == "ResNet50":
		embedding_size = 2048
	else:
		embedding_size = 128

	model_evaluation(model1, train_loader, val_loader, test_loader, plot=True, embedding_size=embedding_size)
	model_evaluation(model2, train_loader, val_loader, test_loader, plot=True, embedding_size=embedding_size)
		

def run_coeval(prediction_only=False):
	if args.input_size == -1:
		# do not resize image. should use with SPP layer
		transforms_args = [transforms.ToTensor(), transforms.Normalize(dataset_mean, dataset_std),]
	else:
		transforms_args = [transforms.Resize((args.input_size, args.input_size)), transforms.ToTensor(), transforms.Normalize(dataset_mean, dataset_std),]
	
	if prediction_only:
		test_dataset = NoLabelFolder(os.path.join(DATA_DIR, args.dataset, "test"),
								transform=transforms.Compose(transforms_args))
	else:
		test_dataset = ImageFolder(os.path.join(DATA_DIR, args.dataset, "test"),
								transform=transforms.Compose(transforms_args))

	test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs) # default

	model1 = load_model(args.backbone, n_classes, False, pt_model_name=args.model1_name, pt_n_classes=args.model1_numclasses)
	model2 = load_model(args.backbone, n_classes, False, pt_model_name=args.model2_name, pt_n_classes=args.model2_numclasses)
	if cuda:
		model1.cuda()
		model2.cuda()
	
	# test
	with torch.no_grad():
		model1.eval()
		model2.eval()

		logit_1 = np.zeros((len(test_loader.dataset), n_classes))
		logit_2 = np.zeros((len(test_loader.dataset), n_classes))
		labels = np.zeros(len(test_loader.dataset))

		k = 0
		for data, target in test_loader:
			if not type(data) in (tuple, list):
				data = (data,)
			if cuda:
				data = tuple(d.cuda() for d in data)

			logit_1[k: k + len(data[0])] = model1(*data).data.cpu().numpy()
			logit_2[k: k + len(data[0])] = model2(*data).data.cpu().numpy()
			labels[k: k + len(data[0])] = target.numpy()

			k += len(data[0])
		logit = np.maximum(logit_1, logit_2)

		if prediction_only:
			sorted_logit_ind = np.argsort(logit, axis=0)[::-1] # sorted in descendent order
			for i, label in enumerate(classes):
				print("--------Top %d samples predicted as %s--------" % (topk, label))
				topk_ind = sorted_logit_ind[:topk, i]

				# visualize top-k prediction
				visualize_images(test_dataset.getfilepath(topk_ind), logit[topk_ind, i],
								os.path.join(RESULT_DIR, "%s_%s_%s.png" % (args.dataset, args.model1_name, label)),
								title=label)
		else:
			print("Prediction of Model 1")
			preds_1 = np.argmax(logit_1, axis=1)
			print(classification_report(labels, preds_1, target_names=classes))
			print(confusion_matrix(labels, preds_1))

			print("Prediction of Model 2")
			preds_2 = np.argmax(logit_2, axis=1)
			print(classification_report(labels, preds_2, target_names=classes))
			print(confusion_matrix(labels, preds_2))

			print("Joint prediction")
			preds = np.argmax(logit, axis=1)
			print(classification_report(labels, preds, target_names=classes))
			print(confusion_matrix(labels, preds))

def run_coteaching():
	augment_transform_args = []
	if args.augment:
		augment_transform_args = [transforms.RandomHorizontalFlip(),
								transforms.RandomVerticalFlip(), transforms.RandomPerspective(),
								transforms.RandomRotation(20)]
	
	transforms_args = [transforms.Resize((args.input_size, args.input_size)), transforms.ToTensor(), transforms.Normalize(dataset_mean, dataset_std)]

	train_dataset = ImageFolder(os.path.join(DATA_DIR, args.dataset, "train"),
							transform=transforms.Compose(augment_transform_args + transforms_args))

	test_dataset = ImageFolder(os.path.join(DATA_DIR, args.dataset, "val"),
							transform=transforms.Compose(transforms_args))

	# define drop rate schedule
	rate_schedule = np.ones(args.n_epoch) * args.keep_rate
	rate_schedule[:args.num_gradual] = np.linspace(1.0, args.keep_rate**args.exponent, args.num_gradual)

	return_embedding = False
	metric_acc = True
	if args.loss_fn.endswith("_triplet"): # metric learning
		print("Learning: Metric learning")
		return_embedding = True # CNN return embedding instead of logit
		metric_acc = False # Do not evaluate accuracy during training
		
	train_batch_sampler = None
	test_batch_sampler = None
	if args.batch_sampler == "balanced":
		print("Sampler: Balanced sampler")
		train_batch_sampler = TrainBalancedBatchSampler(torch.from_numpy(np.array(train_dataset.targets)),
												n_classes=args.train_batch_size // args.n_spc,
												n_batches=args.n_batches,)
		
		test_batch_sampler = TestBalancedBatchSampler(torch.from_numpy(np.array(test_dataset.targets)), 
													n_classes=args.test_batch_size // args.n_spc,
													n_batches=50)
		
	if train_batch_sampler is not None:
		train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
	else:
		train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, **kwargs) # default
	
	if test_batch_sampler is not None:
		test_loader = DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)
	else:
		test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs) # default

	model1 = load_model(args.backbone, n_classes, return_embedding, pt_model_name=args.model1_name, pt_n_classes=args.model1_numclasses)
	model2 = load_model(args.backbone, n_classes, return_embedding, pt_model_name=args.model2_name, pt_n_classes=args.model2_numclasses)
	if cuda:
		model1.cuda()
		model2.cuda()
	
	if args.optim == "Adam":
		optimizer1 = optim.Adam(model1.parameters(), lr=args.lr, weight_decay=1e-4)
		optimizer2 = optim.Adam(model2.parameters(), lr=args.lr, weight_decay=1e-4)
	elif args.optim == "SGD":
		optimizer1 = optim.SGD(model1.parameters(), momentum=0.9, lr=args.lr, weight_decay=1e-4)
		optimizer2 = optim.SGD(model2.parameters(), momentum=0.9, lr=args.lr, weight_decay=1e-4)

	if args.loss_fn == "co_teaching":
		print("Loss fn: CoTeachingLoss")
		loss_fn = CoTeachingLoss(weight=classes_weights, self_taught=args.self_taught, hard_mining=args.hard_mining)
	elif args.loss_fn == "co_teaching_triplet":
		print("Loss fn: CoTeachingTripletLoss")
		loss_fn = CoTeachingTripletLoss(self_taught=args.self_taught, soft_margin=args.soft_margin, hard_mining=args.hard_mining)
	elif args.loss_fn == "co_hardmining_triplet":
		print("Loss fn: CoMiningLoss")		
		loss_fn = CoHardMiningTripletLoss(soft_margin=args.soft_margin)

	lr_scheduler = LrScheduler(args.epoch_decay_start, args.n_epoch, args.lr)

	train_log = []
	for epoch in range(1, args.n_epoch + 1):
		lr_scheduler.adjust_learning_rate(optimizer1, epoch - 1, args.large_batch, args.optim)
		lr_scheduler.adjust_learning_rate(optimizer2, epoch - 1, args.large_batch, args.optim)

		adjust_batch_size(train_batch_sampler, epoch, args.large_batch)

		train_loss_1, train_loss_2, total_train_loss_1, total_train_loss_2 = \
			train_coteaching(train_loader, loss_fn, model1, optimizer1, model2, optimizer2, rate_schedule, epoch, cuda)

		if epoch % args.eval_freq == 0:
			test_loss_1, test_loss_2, test_acc_1, test_acc_2 = \
				eval_coteaching(model1, model2, test_loader, loss_fn, cuda, metric_acc=metric_acc)
			
			train_log.append([train_loss_1, train_loss_2, total_train_loss_1, total_train_loss_2, test_loss_1, test_loss_2])
			print('Epoch [%d/%d], Train loss1: %.4f/%.4f, Train loss2: %.4f/%.4f, Test accuracy1: %.4F, Test accuracy2: %.4f, Test loss1: %.4f, Test loss2: %.4f' 
				% (epoch, args.n_epoch, train_loss_1, total_train_loss_1, train_loss_2, total_train_loss_2, test_acc_1, test_acc_2, test_loss_1, test_loss_2))

			# visualize training log
			train_log_data = np.array(train_log)
			legends = ['train_loss_1', 'train_loss_2', 'total_train_loss_1', 'total_train_loss_2', 'test_loss_1', 'test_loss_2']
			styles = ['b--', 'r--', 'b-.', 'r-.', 'b-', 'r-']
			epoch_count = range(1, train_log_data.shape[0] + 1)
			for i in range(len(legends)):
				plt.loglog(epoch_count, train_log_data[:, i], styles[i])
			plt.legend(legends)
			plt.ylabel('loss')
			plt.xlabel('epochs')
			plt.savefig(os.path.join(MODEL_DIR, '%s_%s_%.2f.png' % (args.dataset, args.loss_fn, args.keep_rate)))
			plt.clf()

		if epoch % args.save_freq == 0:
			torch.save({
						'model_state_dict': model1.state_dict(),
						'optimizer_state_dict': optimizer1.state_dict(),
						'epoch': epoch
						}, os.path.join(MODEL_DIR, '%s_%s_%s_%.2f_1_%d.pth' % (args.dataset, args.backbone, args.loss_fn, args.keep_rate, epoch)))
			
			torch.save({
						'model_state_dict': model2.state_dict(),
						'optimizer_state_dict': optimizer2.state_dict(),
						'epoch': epoch
						}, os.path.join(MODEL_DIR, '%s_%s_%s_%.2f_2_%d.pth' % (args.dataset, args.backbone, args.loss_fn, args.keep_rate, epoch)))

if __name__ == '__main__':
	if args.train:
		run_coteaching()
	elif args.test:
		run_coeval()
	elif args.predict:
		run_coeval_triplet()
		# run_coeval(prediction_only=True)
	else:
		print("Please specify --train, --test, --predict (mutualy exclusive).")
