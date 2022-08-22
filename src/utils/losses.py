import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)

def get_anchor_positive_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    """
    # Check that i and j are distinct
    indices_equal = torch.eye(labels.size(0)).bool()
    indices_not_equal = ~indices_equal

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

    return labels_equal & indices_not_equal


def get_anchor_negative_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)

    return ~(labels.unsqueeze(0) == labels.unsqueeze(1))

def get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    """
    mask_anchor_positive = get_anchor_positive_mask(labels)
    mask_anchor_negative = get_anchor_negative_mask(labels)

    return mask_anchor_positive.unsqueeze(2) & mask_anchor_negative.unsqueeze(1)




def batch_triplet_loss(embeddings, labels, hparams):
    pairwise_dist = torch.cdist(embeddings, embeddings)

    if hparams == 'batch hard':
        ##  for each anchor, select the hardest positive and the hardest negative among the batch
        ## get the hardest positive
        mask_anchor_positive = get_anchor_positive_mask(labels).float()
        anchor_positive_dist = pairwise_dist * mask_anchor_positive
        hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

        ## get the hardest negative
        mask_anchor_negative = get_anchor_negative_mask(labels).float()
        anchor_negative_dist = pairwise_dist + 999. * (1.0 - mask_anchor_negative)
        hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

        triplet_loss = hardest_positive_dist - hardest_negative_dist + 1.0  ## (batch_size, 1)

    elif hparams["mode"] == 'batch all':
        ## select 1) all the valid triplets, and 2) average the loss on the hard and semi-hard triplets
        anchor_positive_dist = pairwise_dist.unsqueeze(2)  ## (batch_size, batch_size, 1)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)  ## (batch_size, 1, batch_size)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + hparams[
            "margin"]  ## broadcasting, all compibations of (a, p, n)
        ## 1) select valid triplets
        valid_mask = get_triplet_mask(labels).float()
        triplet_loss = valid_mask * triplet_loss  ## (batch_size, batch_size, batch_size)
    else:
        raise TypeError("invalid mode")

    # remove easy triplets
    triplet_loss[triplet_loss < 0] = 0
    return torch.mean(triplet_loss)