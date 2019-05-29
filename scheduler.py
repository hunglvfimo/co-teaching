class LrScheduler(object):
	"""docstring for LrScheduler"""
	def __init__(self, epoch_decay_start, n_epoch, lr):
		super(LrScheduler, self).__init__()
		
		self.epoch_decay_start = epoch_decay_start
		self.n_epoch = n_epoch
		self.lr = lr

		self.mom1 = 0.9
		self.mom2 = 0.1

	def adjust_learning_rate(self, optimizer, epoch):
		if epoch < self.epoch_decay_start:
			# lr = self.lr
			beta1 = self.mom1
		else:
			# lr = float(self.n_epoch - epoch) / (self.n_epoch - self.epoch_decay_start) * self.lr
			beta1 = self.mom2

		if epoch == 30 or epoch == 60 or epoch == 80:
			self.lr = self.lr / 10.0

		for param_group in optimizer.param_groups:
			param_group['lr'] = self.lr
			param_group['betas'] = (beta1, 0.999) # Only change beta1


def adjust_batch_size(data_loader, epoch):
	if epoch < 30:
		data_loader.n_samples = 5
		data_loader.n_batches = 2700
	else:
		data_loader.n_samples = 6
		data_loader.n_batches = 1500