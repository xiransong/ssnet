import numpy as np
import torch
import dgl
import os.path as osp
import numba
from tqdm import tqdm
import random


@numba.jit(nopython=True)
def ppr_for_one_node(ptr_array, nei_array, source, num_walks, walk_length, alpha, topk):
    res = numba.typed.Dict.empty(
        key_type=numba.types.int32,
        value_type=numba.types.int32,
    )
    res[source] = 1
    _, degree = ptr_array[source]
    if degree <= 0:
        topk_nids = np.array([[source],[1]], dtype=np.int32)
        return topk_nids
    for _ in range(num_walks):
        u = source
        for _ in range(walk_length):
            ptr, degree = ptr_array[u]
            p = random.random()
            if p > alpha or degree <= 0:
                u = source
                continue
            offset = np.random.randint(low=0, high=degree)
            v = nei_array[ptr + offset]
            if v in res:
                res[v] += 1
            else:
                res[v] = 1
            u = v
    ppr_nids = np.array(list(res.keys()), dtype=np.int32)
    ppr_scores = np.array(list(res.values()), dtype=np.int32)
    
    ind = np.argsort(ppr_scores)[-topk:][::-1]
    topk_nids = ppr_nids[ind]
    topk_ppr_scores = ppr_scores[ind]
    
    re = np.empty(shape=(2, len(topk_nids)), dtype=np.int32)
    re[0] = topk_nids
    re[1] = topk_ppr_scores
    return re


@numba.jit(nopython=True)
def ppr_for_one_node_path2(ptr_array, nei_array, ptr_array_2, nei_array_2, source, num_walks, walk_length, alpha, topk):
    res = numba.typed.Dict.empty(
        key_type=numba.types.int32,
        value_type=numba.types.int32,
    )
    res[source] = 1
    _, degree = ptr_array[source]
    if degree <= 0:
        topk_nids = np.array([[source],[1]], dtype=np.int32)
        return topk_nids
    for hop in range(num_walks):
        u = source
        for hop in range(walk_length):
            if hop == 0:
                _ptr_array = ptr_array
                _nei_array = nei_array
            else:
                _ptr_array = ptr_array_2
                _nei_array = nei_array_2
            ptr, degree = _ptr_array[u]
            p = random.random()
            if p > alpha or degree <= 0:
                u = source
                continue
            offset = np.random.randint(low=0, high=degree)
            v = _nei_array[ptr + offset]
            if v in res:
                res[v] += 1
            else:
                res[v] = 1
            u = v
    ppr_nids = np.array(list(res.keys()), dtype=np.int32)
    ppr_scores = np.array(list(res.values()), dtype=np.int32)
    
    ind = np.argsort(ppr_scores)[-topk:][::-1]
    topk_nids = ppr_nids[ind]
    topk_ppr_scores = ppr_scores[ind]
    
    re = np.empty(shape=(2, len(topk_nids)), dtype=np.int32)
    re[0] = topk_nids
    re[1] = topk_ppr_scores
    return re


@numba.jit(nopython=True, parallel=True)
def ppr_for_batch_nodes_path2(ptr_array, nei_array, ptr_array_2, nei_array_2, batch_nodes, num_walks, walk_length, alpha, topk):
    num_nodes = len(batch_nodes)
    topk_nids_and_ppr_scores_list = [np.zeros(shape=(2,1), dtype=np.int32)] * num_nodes
    for i in numba.prange(num_nodes):
        nid = batch_nodes[i]
        topk_nids_and_ppr_scores = ppr_for_one_node_path2(
            ptr_array, nei_array, ptr_array_2, nei_array_2, nid, num_walks, walk_length, alpha, topk
        )
        topk_nids_and_ppr_scores_list[i] = topk_nids_and_ppr_scores  # do not use list.append !
    return topk_nids_and_ppr_scores_list


# @numba.jit(nopython=True, parallel=True)
# def ppr_for_batch_nodes(ptr_array, nei_array, batch_nodes, num_walks, walk_length, alpha, topk):
#     num_nodes = len(batch_nodes)
#     topk_nids_and_ppr_scores_list = [np.zeros(shape=(2,1), dtype=np.int32)] * num_nodes
#     for i in numba.prange(num_nodes):
#         nid = batch_nodes[i]
#         topk_nids_and_ppr_scores = ppr_for_one_node(
#             ptr_array, nei_array, nid, num_walks, walk_length, alpha, topk
#         )
#         topk_nids_and_ppr_scores_list[i] = topk_nids_and_ppr_scores  # do not use list.append !
#     return topk_nids_and_ppr_scores_list

@numba.jit(nopython=True, parallel=True)
def ppr_for_batch_nodes(ptr_array, nei_array, batch_nodes, num_walks, walk_length, alpha, topk, punish=0):
    num_nodes = len(batch_nodes)
    # topk_nids_and_ppr_scores_list = [np.zeros(shape=(2,1), dtype=np.int32)] * num_nodes
    topk_nids_and_ppr_scores_list = np.zeros(shape=(num_nodes, 3, topk), dtype=np.int32)
    for i in numba.prange(num_nodes):
        nid = batch_nodes[i]
        topk_nids_and_ppr_scores = ppr_for_one_node(
            ptr_array, nei_array, nid, num_walks, walk_length, alpha, topk
        )
        _N = len(topk_nids_and_ppr_scores[0])
        topk_nids_and_ppr_scores_list[i, :2, :_N] = topk_nids_and_ppr_scores[:, :_N]  # do not use list.append !
        topk_nids_and_ppr_scores_list[i, 2, :_N] = 1
    return topk_nids_and_ppr_scores_list
