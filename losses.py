import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from selectors import AllTripletSelector

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float, int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class OnlineTripletLoss(nn.Module):
    def __init__(self, triplet_selector, margin, size_average=True):
        super(OnlineTripletLoss, self).__init__()
        self.triplet_selector = triplet_selector
        self.margin = margin
        self.size_average = size_average

    def forward(self, embeddings, targets):
        triplets = self.triplet_selector.get_triplets(embeddings, targets)
        if embeddings.is_cuda:
            triplets = triplets.cuda()
                
        # calculate distances for pairs
        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)
        loss = F.relu(ap_distances - an_distances + self.margin)
        
        if self.size_average: return loss.mean()
        else: return loss.sum()

class CoTeachingTripletLoss(nn.Module):
    def __init__(self, margin, size_average=True):
        self.margin = margin
        self.size_average = size_average

        super(CoTeachingTripletLoss, self).__init__()

        self.triplet_selector = AllTripletSelector()

    def _triplet_loss(self, emb, triplets):
        ap_distances = (emb[triplets[:, 0]] - emb[triplets[:, 1]]).pow(2).sum(1)
        an_distances = (emb[triplets[:, 0]] - emb[triplets[:, 2]]).pow(2).sum(1)
        loss = F.relu(ap_distances - an_distances + self.margin)
        return loss
    
    def forward(self, emb1, emb2, targets, keep_rate):
        triplets = self.triplet_selector.get_triplets(None, targets)
        if targets.is_cuda:
            triplets = triplets.cuda()

        loss_1 = self._triplet_loss(emb1, triplets)
        ind_1_sorted = np.argsort(loss_1.cpu().data).cuda()

        loss_2 = self._triplet_loss(emb2, triplets)
        ind_2_sorted = np.argsort(loss_2.cpu().data).cuda()

        num_keep = int(keep_rate * len(triplets))

        ind_1_update = ind_1_sorted[:num_keep]
        ind_2_update = ind_2_sorted[:num_keep]

        # exchange samples
        loss_1_update = self._triplet_loss(emb1, triplets[ind_2_update])
        loss_2_update = self._triplet_loss(emb2, triplets[ind_1_update])

        if self.size_average: return loss_1_update.mean(), loss_2_update.mean(), loss_1.mean(), loss_2.mean()
        else: return loss_1_update.sum(), loss_2_update.sum(), loss_1.sum(), loss_2.sum()

class CoTeachingLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        self.weight = weight
        self.size_average = size_average

        super(CoTeachingLoss, self).__init__()
    
    def forward(self, y_1, y_2, targets, keep_rate):
        loss_1 = F.cross_entropy(y_1, targets, weight=self.weight, reduce=False)
        ind_1_sorted = np.argsort(loss_1.cpu().data).cuda()

        loss_2 = F.cross_entropy(y_2, targets, weight=self.weight, reduce=False)
        ind_2_sorted = np.argsort(loss_2.cpu().data).cuda()

        num_keep = int(keep_rate * len(targets))

        ind_1_update = ind_1_sorted[:num_keep]
        ind_2_update = ind_2_sorted[:num_keep]

        # exchange samples
        loss_1_update = F.cross_entropy(y_1[ind_2_update], targets[ind_2_update], weight=self.weight, reduce=False)
        loss_2_update = F.cross_entropy(y_2[ind_1_update], targets[ind_1_update], weight=self.weight, reduce=False)

        if self.size_average: return loss_1_update.mean(), loss_2_update.mean(), loss_1.mean(), loss_2.mean()
        else: return loss_1_update.sum(), loss_2_update.sum(), loss_1.sum(), loss_2.sum()

class CoTeachingLossPlus(nn.Module):
    def __init__(self, weight=None, size_average=True):
        self.weight = weight
        self.size_average = size_average

        super(CoTeachingLossPlus, self).__init__()
    
    def forward(self, y_1, y_2, targets, keep_rate=1.0):
        # select disagreement samples only
        preds_1 = np.argmax(y_1.cpu().data, 1)
        preds_2 = np.argmax(y_2.cpu().data, 1)
        disagreement_mask = (preds_1 != preds_2)
        disagreement_indices = np.where(disagreement_mask)[0]
        disagreement_indices = torch.from_numpy(disagreement_indices).cuda()

        if len(disagreement_indices) == 0:
            # TODO: return zero loss?
            y_1_updated = y_1
            y_2_updated = y_2
            targets_updated = targets
        else:
            y_1_updated = y_1[disagreement_indices]
            y_2_updated = y_2[disagreement_indices]
            targets_updated = targets[disagreement_indices]

        # select small-loss subset from disagreement samples
        loss_1 = F.cross_entropy(y_1_updated, targets_updated, weight=self.weight, reduce=False)
        ind_1_sorted = np.argsort(loss_1.cpu().data).cuda()

        loss_2 = F.cross_entropy(y_2_updated, targets_updated, weight=self.weight, reduce=False)
        ind_2_sorted = np.argsort(loss_2.cpu().data).cuda()

        num_keep = int(keep_rate * len(targets_updated))

        ind_1_update = ind_1_sorted[:num_keep]
        ind_2_update = ind_2_sorted[:num_keep]

        # exchange samples
        loss_1_update = F.cross_entropy(y_1_updated[ind_2_update], targets_updated[ind_2_update], weight=self.weight)
        loss_2_update = F.cross_entropy(y_2_updated[ind_1_update], targets_updated[ind_1_update], weight=self.weight)

        if self.size_average: return loss_1_update.mean(), loss_2_update.mean(), loss_1.mean(), loss_2.mean()
        else: return loss_1_update.sum(), loss_2_update.sum(), loss_1.sum(), loss_2.sum()