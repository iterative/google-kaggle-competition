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




def batch_triplet_loss(embeddings, labels, margin, triplet_type="all"):
    pairwise_dist = torch.cdist(embeddings, embeddings)

    anchor_positive_dist = pairwise_dist.unsqueeze(2)  ## (batch_size, batch_size, 1)
    anchor_negative_dist = pairwise_dist.unsqueeze(1)  ## (batch_size, 1, batch_size)

    triplet_margin = anchor_negative_dist - anchor_positive_dist
    triplet_mask = get_triplet_mask(labels) 
    triplet_loss = -triplet_margin + margin  
    if triplet_type == "semi-hard":

        # loss = max(d_ap - d_an + margin,0) 
        # we want  d_ap - d_an + margin > 0 => d_an - d_ap < margin

        #1st term semi-hard criteria, it means that negative is further from anchor than positive; 
        # 2nd term positive loss; 
        condition = (triplet_margin > 0) & (triplet_margin <= margin) 
        valid_mask = triplet_mask & condition
    else:
        condition = (triplet_margin <= margin) | (triplet_margin > margin)        
       
    valid_mask = triplet_mask & condition
    triplet_loss = triplet_loss[valid_mask]
    triplet_loss[triplet_loss < 0] = 0
    return torch.mean(triplet_loss)
