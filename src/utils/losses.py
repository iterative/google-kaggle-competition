import torch.nn as nn
import torch.nn.functional as F
import torch

def get_anchor_positive_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    """
    # Check that i and j are distinct
    indices_equal = torch.eye(labels.size(0)).bool().to(labels.device)
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




def batch_triplet_loss(embeddings, labels, margin, hardest_only=True):
    pairwise_dist = torch.cdist(embeddings, embeddings)

    if hardest_only:
        mask_anchor_positive = get_anchor_positive_mask(labels).float()
        anchor_positive_dist = pairwise_dist * mask_anchor_positive
        hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

        mask_anchor_negative = get_anchor_negative_mask(labels).float()
        anchor_negative_dist = pairwise_dist + 999. * (1.0 - mask_anchor_negative)
        hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

        triplet_loss = hardest_positive_dist - hardest_negative_dist + margin  ## (batch_size, 1)
    else:
        anchor_positive_dist = pairwise_dist.unsqueeze(2)  ## (batch_size, batch_size, 1)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)  ## (batch_size, 1, batch_size)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin ## broadcasting, all compibations of (a, p, n)
        ## 1) select valid triplets
        valid_mask = get_triplet_mask(labels).float()
        triplet_loss = valid_mask.cuda() * triplet_loss
    # remove easy triplets
    # if not test
    triplet_loss[triplet_loss < 0] = 0
    return torch.mean(triplet_loss)
