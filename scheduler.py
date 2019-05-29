class LrScheduler(object):
	"""docstring for LrScheduler"""
	def __init__(self, epoch_decay_start, n_epoch, lr):
		super(LrScheduler, self).__init__()
		
		self.epoch_decay_start = epoch_decay_start
		self.n_epoch = n_epoch
		self.lr = lr

		self.mom1 = 0.9
		self.mom2 = 0.1

	def adjust_learning_rate(optimizer, epoch):
		if epoch < self.epoch_decay_start:
			# lr = self.lr
			beta1 = self.mom1
		else:
			# lr = float(self.n_epoch - epoch) / (self.n_epoch - self.epoch_decay_start) * self.lr
			beta1 = self.mom2

		if epoch == 30 or epoch == 60 or epoch == 80:
			lr = lr / 10.0

		for param_group in optimizer.param_groups:
			param_group['lr'] = lr
			param_group['betas'] = (beta1, 0.999) # Only change beta1