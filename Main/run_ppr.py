import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from Utils import io
from Utils.parse_arguments import parse_arguments
from Utils.utils import print_dict, ensure_dir
from Model.NonGNN.NumbaPPR import ppr_for_batch_nodes

import numpy as np
import torch
import dgl
import os.path as osp
from tqdm import tqdm
import time
import setproctitle


def get_np_csr_graph(data_root, csr_file=None, dgl_graph_file=None):
    if csr_file is not None:
        csr_file = osp.join(data_root, csr_file)
    else:
        csr_file = osp.join(data_root, "train_graph.np_csr.pkl")
    if osp.exists(csr_file):
        print("load csr graph from cached file")
        csr_graph = io.load_pickle(csr_file)
        ptr_array = csr_graph['ptr_array']    # prt_array: shape=(num_nodes, 2), [[pointer, degree], [pointer, degree], ... ]
        nei_array = csr_graph['nei_array']    # nei_array: shape=(num_edges,)
        num_nodes = len(ptr_array)
    else:
        print("convert dgl graph to csr graph")
        if dgl_graph_file is not None:
            dgl_g = io.load_pickle(osp.join(data_root, dgl_graph_file))
        else:
            dgl_g = io.load_pickle(osp.join(data_root, "train_graph.pkl"))
        num_nodes = dgl_g.num_nodes()
        num_edges = dgl_g.num_edges()
        
        ptr_array = np.empty(shape=(num_nodes, 2), dtype=np.int32)
        nei_array = np.empty(shape=(num_edges,), dtype=np.int32)
        
        ptr_array[:, 1] = dgl_g.out_degrees(dgl_g.nodes()).numpy()
        ptr = 0
        for nid in tqdm(range(num_nodes)):
            ptr_array[nid, 0] = ptr
            degree = ptr_array[nid, 1]
            nei_array[ptr : ptr + degree] = dgl_g.successors(nid).numpy()
            ptr += degree
        
        csr_graph = {'ptr_array': ptr_array, 'nei_array': nei_array}
        io.save_pickle(csr_file, csr_graph)
    
    return num_nodes, ptr_array, nei_array


def main():
    
    parsed_results = parse_arguments()
    '''
    cmd arg requirements:
    --data_root
    --results_root
    --num_walks
    --walk_length
    --alpha
    --topk
    
    '''
    config = parsed_results
    print_dict(config)
    
    data_root = config['data_root']
    results_root = config['results_root']
    ensure_dir(results_root)
    io.save_yaml(osp.join(results_root, "ppr-config.yaml"), config)
    
    num_nodes, ptr_array, nei_array = get_np_csr_graph(
        data_root,
        csr_file="train_undi_graph.np_csr.pkl",
        dgl_graph_file=osp.join(data_root, "train_undi_graph.pkl"))
    
    topk_nids_and_ppr_scores_list = np.zeros(shape=(num_nodes, 3, config['topk']), dtype=np.int32)
    ## dim 0 is neighbor ids
    ## dim 1 is neighbor weight (frequency)
    ## dim 2 is masker,  1 or 0
    
    print("run ppr...")
    nodes_dl = torch.utils.data.DataLoader(dataset=torch.arange(num_nodes), batch_size=512)
    idx_start = 0
    for nids in tqdm(nodes_dl):
        _N = len(nids)
        topk_nids_and_topk_ppr_scores = ppr_for_batch_nodes(
            ptr_array=ptr_array, nei_array=nei_array,
            batch_nodes=nids.numpy(),
            num_walks=config['num_walks'],
            walk_length=config['walk_length'],
            alpha=config['alpha'],
            topk=config['topk'])
        topk_nids_and_ppr_scores_list[idx_start:idx_start+_N] = topk_nids_and_topk_ppr_scores[:]
        idx_start += _N
    
    print("saving model outputs...")
    # io.save_pickle(osp.join(results_root, "topk_nids_and_ppr_scores_list.pkl"), topk_nids_and_ppr_scores_list)
    
    nei = topk_nids_and_ppr_scores_list[:, 0, :] #np.zeros((_N, topk), dtype=np.int32)
    wei = topk_nids_and_ppr_scores_list[:, 1, :] # np.zeros((_N, topk), dtype=np.int32)
    wei_uniform = topk_nids_and_ppr_scores_list[:, 2, :]
    
    io.save_pickle(osp.join(results_root, "nei.pkl"), nei)  # ppr neighbors
    io.save_pickle(osp.join(results_root, "wei.pkl"), wei)
    io.save_pickle(osp.join(results_root, "wei_uniform.pkl"), wei_uniform)


if __name__ == "__main__":
    
    setproctitle.setproctitle('ppr-' + 
                              time.strftime("%m%d-%H%M%S", time.localtime(time.time())))
    
    main()
