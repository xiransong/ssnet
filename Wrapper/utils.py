from Utils import io

import torch
import os.path as osp


def load_appr_neighbors_and_weights(appr_data_root, topk):
    raw_nei = io.load_pickle(osp.join(appr_data_root, "appr_neighbors-padded.pkl"))
    raw_wei = io.load_pickle(osp.join(appr_data_root, "appr_weights-padded.pkl"))
    
    nei = torch.LongTensor(raw_nei[:, -topk - 1: -1])  # don't use self
    wei = torch.FloatTensor(raw_wei[:, -topk - 1: -1])
    wei = torch.softmax(wei, dim=-1)
    
    return nei, wei


def aggregate_by_weights(nids, nei, wei, base_emb_table, gnn_forward_device):
    top_nids = nei[nids]

    top_weights = wei[nids]

    top_embs = base_emb_table[top_nids]

    top_weights = top_weights.to(gnn_forward_device)

    top_embs = top_embs.to(gnn_forward_device)
        
    out_embs = torch.matmul(top_weights.unsqueeze(-2), top_embs)
        
    return out_embs.squeeze()
