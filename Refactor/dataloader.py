from Utils import io
import Utils.utils
import Module.gnn as gnn_module

import numpy as np
import numba
import torch
import dgl
import dgl.dataloading as dgldl
from tqdm import tqdm
import os.path as osp


@numba.jit(nopython=True)
def get_csr_graph_neighbors(ptr_array, nei_array, u):
    ptr, degree = ptr_array[u]
    return nei_array[ptr : ptr + degree]


@numba.jit(nopython=True, parallel=True)
def sample_a_neighbor(src, ptr_array, nei_array):
    dst = np.empty(len(src), dtype=np.int32)
    for i in numba.prange(len(src)):
        nei = get_csr_graph_neighbors(ptr_array, nei_array, src[i])
        if len(nei) == 0:
            dst[i] = src[i]
        else:
            dst[i] = np.random.choice(nei, 1, replace=False)[0]
    return dst


class NodeBased_EdgeSampler:
    
    def __init__(self, batch_size, ptr_array, nei_array, 
                 weighted_sampling=False, alpha=None):
        self.batch_size = batch_size
        self.num_nodes = len(ptr_array)
        self.ptr_array = ptr_array
        self.nei_array = nei_array
        self.src_dl = torch.utils.data.DataLoader(
            torch.arange(self.num_nodes), batch_size=batch_size, shuffle=True
        )
        self.src_dl_iter = None
        self.weighted_sampling = weighted_sampling
        if self.weighted_sampling:
            print("## node-based degree weighted sampling")
            degrees = torch.FloatTensor(ptr_array[:,1])
            self.weights = degrees ** alpha
        
    def __len__(self):
        return len(self.src_dl)
    
    def __iter__(self):
        self.src_dl_iter = iter(self.src_dl)
        return self
    
    def __next__(self):
        src = next(self.src_dl_iter)
        
        if self.weighted_sampling:
            src = torch.multinomial(self.weights, num_samples=len(src),
                                    replacement=True)
        
        pos = sample_a_neighbor(src.numpy(), self.ptr_array, self.nei_array)
        pos = torch.LongTensor(pos)
        
        neg = torch.randint(0, self.num_nodes, (len(src),))
        mask = src == pos
        neg[mask] = src[mask]
        
        return torch.cat([src, pos, neg])
    
    def __getitem__(self, idx):
        if self.src_dl_iter is None:
            self.src_dl_iter = iter(self.src_dl)
        try:
            return self.__next__()
        except StopIteration:
            self.src_dl_iter = iter(self.src_dl)
            return self.__next__()
    
    
class EdgeBased_EdgeSampler(torch.utils.data.Dataset):
    
    def __init__(self, num_nodes, batch_size, E_src, E_dst):
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.batch_per_epoch = int(np.ceil(num_nodes / batch_size))  # in each epoch, sample num_nodes edges
        self.num_edges = len(E_src)
        self.E_src = E_src
        self.E_dst = E_dst
    
    def __len__(self):
        return self.batch_per_epoch
    
    def __getitem__(self, idx):
        # ignore idx
        eid = torch.randint(0, self.num_edges, (self.batch_size,))  # select an edge from all edges
        src = self.E_src[eid]
        pos = self.E_dst[eid]
        neg = torch.randint(0, self.num_nodes, (self.batch_size,))
        return torch.cat([src, pos, neg])


class BlockTrainDataLoader:
    
    def __init__(self):
        pass
    
    def build(self, data, config):
        # init edge_sampler
        train_graph = io.load_pickle(osp.join(config['data_root'], "train_graph.pkl"))
        
        data['in_degrees'] = train_graph.in_degrees()
        data['num_nodes'] = train_graph.num_nodes()
        
        if config['use_nodebased_dl']:
            print("## using node-based dataloader")
            csr_graph = io.load_pickle(osp.join(config['data_root'], "train_graph.np_csr.pkl"))
            edge_sampler = NodeBased_EdgeSampler(
                config['train_batch_size'], 
                csr_graph['ptr_array'], 
                csr_graph['nei_array'],
                weighted_sampling=config['use_weighted_sampling'], 
                alpha=config['alpha'])
        else:
            E = train_graph.edges()
            num_nodes, E_src, E_dst = train_graph.num_nodes(), E[0], E[1]
            edge_sampler = EdgeBased_EdgeSampler(num_nodes, config['train_batch_size'], E_src, E_dst)
        
        del train_graph
        
        # init node_collator
        if config['model'] not in {'pprgo', 'ganpprgo'}:
            node_collate_graph = io.load_pickle(osp.join(config['data_root'], "train_undi_graph.pkl"))
            if config['model'] in {'lightgcn', 'ganlightgcn', 'sagn', 'fagcn'}:
                full_adj = gnn_module.add_edge_weights_to_graph(node_collate_graph, return_adj_matrix=True)
                data['full_adj'] = full_adj
            
            if config['model'] in {'graphsage', 'gat', 'gin', 'sagn'} and config['layer_sample']:
                block_sampler = dgldl.MultiLayerNeighborSampler(
                    eval(config['num_layer_sample'])  # e.g. [10, 10]
                )
            else:
                block_sampler = dgldl.MultiLayerFullNeighborSampler(
                    config['num_gcn_layer']
                )
            
            self.node_collator = dgldl.NodeCollator(
                g=node_collate_graph,
                nids=node_collate_graph.nodes(),
                block_sampler=block_sampler
            )
            data['node_collator'] = self.node_collator
        
        # build torch DataLoader by edge_sampler and collate_fn
        if config['src_pos_neg_collate'] or config['model'] in {'pprgo', 'ganpprgo'}:
            
            def collate_fn(batch_nids): 
                batch_nids = batch_nids[0].view(3, -1)
                src, pos, neg = batch_nids[0, :], batch_nids[1, :], batch_nids[2, :]
                return src, pos, neg
        
        elif config['model'] in {'lightgcn', 'ganlightgcn', 'sagn'} and config['num_gcn_layer'] == 2:
            
            self.node_collator1 = dgldl.NodeCollator(
                g=node_collate_graph,
                nids=node_collate_graph.nodes(),
                block_sampler=dgldl.MultiLayerFullNeighborSampler(1)
            )
            
            def collate_fn(batch_nids):
                batch_nids = batch_nids[0]
                unique_batch_nids = list(set(batch_nids.tolist()))
                
                nid_mapping = {nid: i for i, nid in enumerate(unique_batch_nids)}
                
                def index_mapping(nid):
                    return nid_mapping[nid]

                def handle_idx(nids):
                    return Utils.utils.element_wise_map(index_mapping, nids)
                
                local_idx = handle_idx(batch_nids)
                
                B1 = self.node_collator1.collate(unique_batch_nids)
                B2 = self.node_collator.collate(unique_batch_nids)
                
                return batch_nids, local_idx, B1, B2
            
        else:
            
            def collate_fn(batch_nids):
                batch_nids = batch_nids[0]
                unique_batch_nids = list(set(batch_nids.tolist()))
                
                nid_mapping = {nid: i for i, nid in enumerate(unique_batch_nids)}
                
                def index_mapping(nid):
                    return nid_mapping[nid]

                def handle_idx(nids):
                    return Utils.utils.element_wise_map(index_mapping, nids)
                
                local_idx = handle_idx(batch_nids)
                
                return batch_nids, local_idx, *(self.node_collator.collate(unique_batch_nids))
        
        self.dl = torch.utils.data.DataLoader(
            dataset=edge_sampler,
            batch_size=1,
            collate_fn=collate_fn,
            num_workers=config['num_workers']  # ``num_workers=0`` means that the data will be loaded in the main process
        )
        
    def __iter__(self):
        return iter(tqdm(self.dl))


class ClusterGCNDataLoader:

    def __init__(self):
        pass
    
    def build(self, data, config):
        self.train_graph = io.load_pickle(osp.join(config['data_root'], "train_graph.pkl"))
        self.train_undi_graph = io.load_pickle(osp.join(config['data_root'], "train_undi_graph.pkl"))
        data['num_nodes'] = self.train_undi_graph.num_nodes()
        
        full_adj = gnn_module.add_edge_weights_to_graph(self.train_undi_graph, return_adj_matrix=True)
        data['full_adj'] = full_adj
        self.edge_weights = self.train_undi_graph.edata['ew']
        
        self.node_groups = io.load_pickle(config['file_node_groups'])
        num_cluster = len(self.node_groups)
        num_combined_cluster=config['num_combined_cluster']
        self.cid_dl = torch.utils.data.DataLoader(dataset=torch.arange(num_cluster),
                                                  batch_size=num_combined_cluster,
                                                  shuffle=True)
        self.train_batch_size = config['train_batch_size']
        self.num_workers = config['num_workers']
    
    def _gen_batch(self):
        for cids in tqdm(self.cid_dl):
            subgraph_nids = torch.cat([torch.tensor(self.node_groups[cid]) for cid in cids.numpy()])
            
            # build subgraph adjacency matrix
            train_undi_subgraph = dgl.node_subgraph(self.train_undi_graph, subgraph_nids)
            idx = torch.stack(train_undi_subgraph.edges())
            subgraph_edge_weights = self.edge_weights[train_undi_subgraph.edata[dgl.EID]]
            num_subgraph_nodes = train_undi_subgraph.num_nodes()
            sub_adj = torch.sparse_coo_tensor(idx, subgraph_edge_weights,
                                              (num_subgraph_nodes, num_subgraph_nodes))            
            
            # build subgraph train_dl
            train_subgraph = dgl.node_subgraph(self.train_graph, subgraph_nids)
            E = train_subgraph.edges()
            num_nodes, E_src, E_dst = train_subgraph.num_nodes(), E[0], E[1]
            edge_sampler = EdgeBased_EdgeSampler(num_nodes, self.train_batch_size, E_src, E_dst)
            
            sub_train_dl = torch.utils.data.DataLoader(
                dataset=edge_sampler,
                batch_size=1,
            )
            
            for b in sub_train_dl:
                b = b.view(3, -1)
                src, pos, neg = b[0], b[1], b[2]
                yield subgraph_nids, sub_adj, src, pos, neg
        
    def __iter__(self):
        self.it = self._gen_batch()
        return self
    
    def __len__(self):
        return len(self.cid_dl)
    
    def __next__(self):
        return next(self.it)
