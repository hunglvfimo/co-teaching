This repository is forked from https://github.com/bhanML/Co-teaching

Original paper: https://arxiv.org/abs/1804.06872

Co-teaching is originaly developed for classification task in noisy data. We adapt the idea of co-teaching for metric learning task.

Popularly, metric learning model involves the selection of pair (or triplet) sample using algorithms such as HardBatch, SoftBatch, ... The popular idea is to select a subset of pair (or triplet) samples in each mini-batch at each stage so that the model can learn better. 

Using co-teaching, two metric learning models (i.e. Siamese, Triplet) are trained simultaneously. In each mini-batch, both models are fetched the same set of samples. Each model will then select a subset of pair (or triplet) samples which are hardest for them. The selected samples of model A will be used to calculate loss and update gradient for model B, and vice versa.
