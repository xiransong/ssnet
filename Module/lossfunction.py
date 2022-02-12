import torch
import torch.nn.functional as F


def bpr_loss(pos_score, neg_score):
    return torch.mean(F.softplus(neg_score - pos_score))


def neg_loss(pos_score, neg_score):
    
    if pos_score.flatten().shape == neg_score.flatten().shape:
        return torch.mean(F.softplus(neg_score - pos_score))
    else:
        return torch.mean(F.softplus(neg_score.exp().sum(dim=-1).log() - pos_score))
