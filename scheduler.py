def adjust_learning_rate(optimizer, alpha_plan, beta1_plan, epoch):
	for param_group in optimizer.param_groups:
		param_group['lr']=alpha_plan[epoch]
		param_group['betas']=(beta1_plan[epoch], 0.999) # Only change beta1