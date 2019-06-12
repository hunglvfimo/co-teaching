import torch.nn as nn
from torch.utils.data.sampler import BatchSampler

import numpy as np

class TrainBalancedBatchSampler(BatchSampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        labels (Tensor): list of labels
        n_samples (int): number of sample of each class in each batch
        n_classes (int): number of class label for each batch. batch_size = n_samples * n_classes
    """

    def __init__(self, labels, n_classes=8, n_batches=100):
        np.random.seed(1)
        
        self.n_samples = 4
        self.n_classes = n_classes
        self.n_batches = n_batches
               
        self.labels = labels.numpy()
        self.labels_set = list(set(self.labels)) # list labels for samples
        self.label_to_indices = {label: np.where(self.labels == label)[0] 
                                    for label in self.labels_set} 

    def __iter__(self):
        # shuffle dataset after each epoch
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l]) 
        
        used_label_indices_count = {label: 0 for label in self.labels_set}        
        for batch in range(self.n_batches):
            indices = []
            for class_ in np.random.choice(self.labels_set, size=self.n_classes, replace=False):
                # select next n_samples from each class
                indices.extend(self.label_to_indices[class_][used_label_indices_count[class_]: used_label_indices_count[class_] + self.n_samples])
                used_label_indices_count[class_] += self.n_samples
                
                if used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    used_label_indices_count[class_] = 0
            
            yield indices

    def __len__(self):
        return self.n_batches

class TestBalancedBatchSampler(BatchSampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        labels (Tensor): list of labels
        n_samples (int): number of sample of each class in each batch
        n_classes (int): number of class label for each batch. batch_size = n_samples * n_classes
    """

    def __init__(self, labels, n_classes=8, n_batches=100):
        np.random.seed(1)
        
        self.n_samples = 2
        self.n_classes = n_classes
        self.n_batches = n_batches
                
        self.labels = labels.numpy()
        self.labels_set = list(set(self.labels)) # list labels for samples
        self.label_to_indices = {label: np.where(self.labels == label)[0] 
                                    for label in self.labels_set}
        self._generate_batches()

    def _generate_batches(self):
        used_label_indices_count = {label: 0 for label in self.labels_set}

        self.batches = []
        for batch in range(self.n_batches):
            indices = []
            for class_ in np.random.choice(self.labels_set, size=self.n_classes, replace=False):
                # select next n_samples from each class
                indices.extend(self.label_to_indices[class_][used_label_indices_count[class_]: used_label_indices_count[class_] + self.n_samples])
                used_label_indices_count[class_] += self.n_samples
                
                if used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    used_label_indices_count[class_] = 0
            
            self.batches.append(indices)

    def __iter__(self):
        for batch in range(self.n_batches):
            yield self.batches[batch]

    def __len__(self):
        return self.n_batches