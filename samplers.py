import torch.nn as nn
from torch.utils.data.sampler import BatchSampler

import numpy as np

class BalancedBatchSampler(BatchSampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        labels (Tensor): list of labels
        n_samples (int): number of sample of each class in each batch
    """

    def __init__(self, labels, n_samples=8, n_batches=100, training=True):
        np.random.seed(2528)
        
        self.n_samples = n_samples
        self.n_batches = n_batches
        self.training = training
        
        self.labels = labels.numpy()
        self.labels_set = list(set(self.labels)) # list labels for samples
        self.label_to_indices = {label: np.where(self.labels == label)[0] 
                                    for label in self.labels_set} 

    def __iter__(self):
        # shuffle dataset after each epoch
        if self.training:
            for l in self.labels_set:
                np.random.shuffle(self.label_to_indices[l]) 
        
        used_label_indices_count = {label: 0 for label in self.labels_set}        
        for batch in range(self.n_batches):
            indices = []
            for class_ in self.labels_set:
                # select next n_samples from each class
                indices.extend(self.label_to_indices[class_][used_label_indices_count[class_]: used_label_indices_count[class_] + self.n_samples])
                used_label_indices_count[class_] += self.n_samples
                
                if used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    used_label_indices_count[class_] = 0
            
            yield indices

    def __len__(self):
        return self.n_batches