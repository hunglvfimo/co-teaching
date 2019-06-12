import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from selectors import AllTripletSelector, HardestNegativeTripletSelector

class CoTeachingLoss(nn.Module):
    def __init__(self, weight=None, self_taught=False, hard_mining=False, size_average=True):
        self.hard_mining = hard_mining
        self.self_taught = self_taught
        self.weight = weight
        self.size_average = size_average

        super(CoTeachingLoss, self).__init__()
    
    def forward(self, y_1, y_2, targets, keep_rate):
        loss_1 = F.cross_entropy(y_1, targets, weight=self.weight, reduce=False)
        loss_2 = F.cross_entropy(y_2, targets, weight=self.weight, reduce=False)

        loss_1_update = loss_1
        loss_2_update = loss_2
        if keep_rate < 1.0:
            ind_1_sorted = np.argsort(loss_1.cpu().data.numpy())
            ind_2_sorted = np.argsort(loss_2.cpu().data.numpy())
            if self.hard_mining:
                ind_1_sorted = ind_1_sorted[::-1]
                ind_2_sorted = ind_2_sorted[::-1]

            ind_1_sorted = torch.LongTensor(ind_1_sorted.copy()).cuda()
            ind_2_sorted = torch.LongTensor(ind_2_sorted.copy()).cuda()

            num_keep = int(keep_rate * len(targets))

            if self.self_taught:
                ind_1_update = ind_1_sorted[:num_keep]
                ind_2_update = ind_2_sorted[:num_keep]
            else:
                ind_1_update = ind_2_sorted[:num_keep]
                ind_2_update = ind_1_sorted[:num_keep]

            # exchange samples
            loss_1_update = F.cross_entropy(y_1[ind_1_update], targets[ind_1_update], weight=self.weight, reduce=False)
            loss_2_update = F.cross_entropy(y_2[ind_2_update], targets[ind_2_update], weight=self.weight, reduce=False)

        if self.size_average: return loss_1_update.mean(), loss_2_update.mean(), loss_1.mean(), loss_2.mean()
        else: return loss_1_update.sum(), loss_2_update.sum(), loss_1.sum(), loss_2.sum()

class CoHardMiningTripletLoss(nn.Module):
    def __init__(self, soft_margin, size_average=True):
        self.soft_margin = soft_margin
        self.size_average = size_average

        super(CoHardMiningTripletLoss, self).__init__()

        self.hard_batch = HardestNegativeTripletSelector(soft_margin=self.soft_margin)

    def _triplet_loss(self, emb, triplets):
        ap_distances = (emb[triplets[:, 0]] - emb[triplets[:, 1]]).pow(2).sum(1)
        an_distances = (emb[triplets[:, 0]] - emb[triplets[:, 2]]).pow(2).sum(1)
        
        if self.soft_margin:
            loss = F.softplus(ap_distances - an_distances)
        else:
            loss = F.relu(ap_distances - an_distances + 0.1)
        return loss
    
    def forward(self, emb1, emb2, targets, keep_rate):
        hard_triplets1 = self.hard_batch.get_triplets(emb1, targets)
        hard_triplets2 = self.hard_batch.get_triplets(emb2, targets)
        if targets.is_cuda:
            hard_triplets1 = hard_triplets1.cuda()
            hard_triplets2 = hard_triplets2.cuda()

        hard_loss_1 = self._triplet_loss(emb1, hard_triplets2)
        hard_loss_2 = self._triplet_loss(emb2, hard_triplets1)

        if self.size_average: return hard_loss_1.mean(), hard_loss_2.mean(), hard_loss_1.mean(), hard_loss_2.mean()
        else: return hard_loss_1.sum(), hard_loss_2.sum(), hard_loss_1.sum(), hard_loss_2.sum()

class CoTeachingTripletLoss(nn.Module):
    def __init__(self, self_taught=False, soft_margin=False, hard_mining=False, size_average=True):
        self.self_taught = self_taught
        self.hard_mining = hard_mining
        self.soft_margin = soft_margin
        self.size_average = size_average

        super(CoTeachingTripletLoss, self).__init__()

        self.all_batch = AllTripletSelector()

    def _triplet_loss(self, emb, triplets):
        ap_distances = (emb[triplets[:, 0]] - emb[triplets[:, 1]]).pow(2).sum(1)
        an_distances = (emb[triplets[:, 0]] - emb[triplets[:, 2]]).pow(2).sum(1)
        
        if self.soft_margin:
            loss = F.softplus(ap_distances - an_distances)
        else:
            loss = F.relu(ap_distances - an_distances + 0.1)
        return loss
    
    def forward(self, emb1, emb2, targets, keep_rate):
        all_triplet = self.all_batch.get_triplets(None, targets)        
        if targets.is_cuda:
            all_triplet = all_triplet.cuda()

        loss_1 = self._triplet_loss(emb1, all_triplet)
        loss_2 = self._triplet_loss(emb2, all_triplet)

        if keep_rate < 1.0:
            ind_1_sorted = np.argsort(loss_1.cpu().data.numpy())
            ind_2_sorted = np.argsort(loss_2.cpu().data.numpy())
            if self.hard_mining:
                ind_1_sorted = ind_1_sorted[::-1]
                ind_2_sorted = ind_2_sorted[::-1]

            ind_1_sorted = torch.LongTensor(ind_1_sorted.copy()).cuda()
            ind_2_sorted = torch.LongTensor(ind_2_sorted.copy()).cuda()

            num_keep = int(keep_rate * len(all_triplet))

            if self.self_taught:
                ind_1_update = ind_1_sorted[:num_keep]
                ind_2_update = ind_2_sorted[:num_keep]
            else:
                ind_1_update = ind_2_sorted[:num_keep]
                ind_2_update = ind_1_sorted[:num_keep]

            # exchange samples
            loss_1_update = self._triplet_loss(emb1, all_triplet[ind_2_update])
            loss_2_update = self._triplet_loss(emb2, all_triplet[ind_1_update])
        else:
            if self.self_taught:
                loss_1_update = loss_1
                loss_2_update = loss_2
            else:
                loss_1_update = loss_2
                loss_2_update = loss_1

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