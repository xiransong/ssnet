import torch
import dgl
from tqdm import tqdm
import Utils.utils
import Utils.metric
from Utils.Timer import Timer


def _get_gnn_output_emb(output_nids,
                        base_emb_table, gnn, gnn_forward_device, node_collator,
                        return_input_emb=False, remove_pos_edge=False, is_reverse_graph=False, src=None, dst=None):
    """Get gnn's output embeddings by specifying corresponding nodes.
    
    Args:
        output_nids         (list): nodes' IDs.
        base_emb_table      (torch.Tensor): input embedding table.
        gnn                 (torch.nn.Module): gnn.
        gnn_forward_device          (str): gnn's device.
        node_collator       (dgl.dataloading.NodeCollator): dgl NodeCollator.
        return_input_emb    (:obj:`bool`, optional): if True, return the input embeddings. defaults to False.
    """
    # timer = Timer()
    
    # timer.start("collate")
    input_nids, _, blocks = node_collator.collate(output_nids)
    # timer.end("collate")
    
    input_emb = base_emb_table[input_nids]
    
    input_emb = input_emb.to(gnn_forward_device)
    blocks = [block.to(gnn_forward_device) for block in blocks]
    
    if remove_pos_edge:
        assert len(blocks) == 1, "currently, remove_pos_edge only supports 1 layer gnn"
        block = blocks[0]
        _nid = block.ndata[dgl.NID]['_N']
        device = _nid.device
        _nid = _nid.cpu().numpy()
        nid_mapping = {_nid[i]:i for i in range(len(_nid))}
        
        _src = []
        _dst = []
        for k in range(len(src)):
            if src[k] in nid_mapping and dst[k] in nid_mapping:
                _src.append(nid_mapping[src[k]])
                _dst.append(nid_mapping[dst[k]])
        _src = torch.LongTensor(_src).to(device)
        _dst = torch.LongTensor(_dst).to(device)
        
        if is_reverse_graph:
            mask = block.has_edges_between(_dst, _src)
            eids = block.edge_ids(_dst[mask], _src[mask])
            block.remove_edges(eids)
        else:
            mask = block.has_edges_between(_src, _dst)
            eids = block.edge_ids(_src[mask], _dst[mask])
            block.remove_edges(eids)
    # timer.start("gnn")
    output_emb = gnn(blocks, input_emb)
    # timer.end("gnn")
    # print(timer.get_all_mean_time())
    
    # import pdb; pdb.set_trace()
    if return_input_emb:
        return output_emb, input_emb
    else:
        return output_emb


def get_gnn_output_emb(base_emb_table, gnn, gnn_forward_device, node_collator, 
                       nids_list: list,
                       remove_pos_edge=False, is_reverse_graph=False, src=None, dst=None):
    """Get gnn's output embeddings by specifying corresponding nodes. Keep the tensors' shape."""
    
    output_nids = list(set(
        torch.cat([nids.flatten() for nids in nids_list]).numpy()
    ))
    
    output_emb = _get_gnn_output_emb(output_nids, base_emb_table, gnn, gnn_forward_device, 
                                     node_collator, 
                                     remove_pos_edge=remove_pos_edge, is_reverse_graph=is_reverse_graph, 
                                     src=src, dst=dst)
        
    nid_mapping = {nid: i for i, nid in enumerate(output_nids)}

    def index_mapping(nid):
        return nid_mapping[nid]

    def handle_idx(nids):
        return Utils.utils.element_wise_map(index_mapping, nids)

    idx_list = [handle_idx(nids) for nids in nids_list]
    
    emb_list = [output_emb[idx] for idx in idx_list]
    
    return emb_list


def dot_prodoct_score(src_emb, dst_emb, neg_dst_emb):
    pos_score = torch.sum(src_emb * dst_emb, dim=-1)
    
    if src_emb.shape != neg_dst_emb.shape:
        src_emb = torch.repeat_interleave(
            src_emb, neg_dst_emb.shape[-2], dim=-2
        ).reshape(neg_dst_emb.shape)
        
    neg_score = torch.sum(src_emb * neg_dst_emb, dim=-1)
    
    return pos_score, neg_score


def get_all_gnn_output_emb(base_emb_table, gnn, gnn_forward_device, out_emb_table, 
                           batch_size, node_collator):
    all_nids = torch.arange(len(base_emb_table))
    nodes_dl = torch.utils.data.DataLoader(dataset=all_nids, batch_size=batch_size)
    
    for output_nids in tqdm(nodes_dl, desc="get all gnn output embs"):
        output_emb = _get_gnn_output_emb(output_nids.tolist(), 
                                        base_emb_table, gnn, gnn_forward_device, node_collator, 
                                        return_input_emb=False)
        out_emb_table[output_nids] = output_emb.to(out_emb_table.device)


def eval_by_out_emb_table(out_emb_table, dl):
    S = []
    for src, dst, neg_dst in tqdm(dl, desc="eval"):
        src_emb, dst_emb, neg_dst_emb = map(lambda idx: out_emb_table[idx],
                                            [src, dst, neg_dst])
        
        pos_score, neg_score = dot_prodoct_score(src_emb, dst_emb, neg_dst_emb)
        
        pos_neg_score = torch.cat((pos_score.view(-1, 1), neg_score), dim=-1).detach().cpu()
        S.append(pos_neg_score)
    
    S = torch.cat(S, dim=0).numpy()
    results = Utils.metric.all_metrics(S)
    return results


class TwoLayerMLP(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.m = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
        )
    
    def forward(self, x):
        return self.m(x)


class SAGE(torch.nn.Module):
    
    def __init__(
            self, 
            arch=[
                {"in_feats": 64, "out_feats": 64, "aggregator_type": 'pool', "activation": torch.tanh},
                {"in_feats": 64, "out_feats": 64, "aggregator_type": 'pool'}
                ],
            mlp_list=None
        ):
        super().__init__()
        self.gnn_layers = torch.nn.ModuleList([
            dgl.nn.SAGEConv(**layer_arch) for layer_arch in arch
        ])
        self.mlp_list = mlp_list

    def forward(self, blocks, x):
        assert len(blocks) == len(self.gnn_layers)
        if self.mlp_list is not None:
            x = self.mlp_list[0](x)
        for i, (block, gnn_layer) in enumerate(zip(blocks, self.gnn_layers)):
            x = gnn_layer(block, x)
            if self.mlp_list is not None:
                x = self.mlp_list[i + 1](x)
        return x


class GAT(torch.nn.Module):
    
    def __init__(
            self, 
            arch=[
                {"in_feats": 64, "out_feats": 64, "num_heads": 4, "activation": torch.tanh},
                {"in_feats": 64, "out_feats": 64, "num_heads": 4}
                ],
            mlp_list=None
        ):
        super().__init__()
        self.gnn_layers = torch.nn.ModuleList([
            dgl.nn.GATConv(**layer_arch) for layer_arch in arch
        ])
        self.mlp_list = mlp_list
    
    def forward(self, blocks, x):
        assert len(blocks) == len(self.gnn_layers)
        if self.mlp_list is not None:
            x = self.mlp_list[0](x)
        for i, (block, gnn_layer) in enumerate(zip(blocks, self.gnn_layers)):
            x = gnn_layer(block, x)
            x = x.mean(dim=-2)  # merge the outputs of different heads
            if self.mlp_list is not None:
                x = self.mlp_list[i + 1](x)
        return x


class GIN(torch.nn.Module):
    
    def __init__(
            self, 
            num_gcn_layer,
            mlp_list=None
        ):
        super().__init__()
        
        self.gin_mlp_list = torch.nn.ModuleList([TwoLayerMLP() for _ in range(num_gcn_layer)])
        
        self.gnn_layers = torch.nn.ModuleList([
            dgl.nn.GINConv(apply_func=self.gin_mlp_list[i], aggregator_type="sum") 
            for i in range(num_gcn_layer)
        ])
        self.mlp_list = mlp_list
    
    def forward(self, blocks, x):
        assert len(blocks) == len(self.gnn_layers)
        if self.mlp_list is not None:
            x = self.mlp_list[0](x)
        for i, (block, gnn_layer) in enumerate(zip(blocks, self.gnn_layers)):
            x = gnn_layer(block, x)
            if self.mlp_list is not None:
                x = self.mlp_list[i + 1](x)
        return x
        

class LightGCN(torch.nn.Module):
    
    def __init__(self, mlp_list=None):
        super().__init__()
        self.gcn_msg = dgl.function.u_mul_e('h', 'ew', 'm')
        self.gcn_reduce = dgl.function.sum(msg='m', out='h')
        self.mlp_list = mlp_list
    
    def forward(self, blocks, x):
        if self.mlp_list is not None:
            x = self.mlp_list[0](x)
        for i in range(len(blocks)):
            blocks[i].srcdata['h'] = x
            blocks[i].update_all(self.gcn_msg, self.gcn_reduce)
            x = blocks[i].dstdata['h']
            if self.mlp_list is not None:
                x = self.mlp_list[i](x)
        return x


class FAGCN(torch.nn.Module):
    
    def __init__(self, emb_dim):
        super().__init__()
        self.linear_src = torch.nn.Linear(emb_dim, 1, bias=False)
        self.linear_dst = torch.nn.Linear(emb_dim, 1, bias=False)

        self.gcn_msg = dgl.function.u_mul_e('h', 'a', 'm')
        self.gcn_reduce = dgl.function.sum(msg='m', out='h')
        self.tanh = torch.nn.Tanh()
    
    def forward(self, blocks, x):
        for g in blocks:
            with g.local_scope():
                # import pdb; pdb.set_trace()
                # calc attention weight
                g.srcdata['h'] = x
                
                g.srcdata['a_src'] = self.linear_src(x[g.srcnodes()]).squeeze()
                g.dstdata['a_dst'] = self.linear_dst(x[g.dstnodes()]).squeeze()
                g.apply_edges(dgl.function.u_add_v('a_src', 'a_dst', 'a'))
                
                g.edata['a'] = self.tanh(g.edata['a']) * g.edata['ew']  # the final edge weight
                
                g.update_all(self.gcn_msg, self.gcn_reduce)
                x = g.dstdata['h']
        return x


def add_edge_weights_to_graph(dgl_graph, return_adj_matrix=False):
    E = dgl_graph.edges()
    d1 = (dgl_graph.out_degrees(E[0]) + dgl_graph.in_degrees(E[0])) / 2.0
    d2 = (dgl_graph.out_degrees(E[1]) + dgl_graph.in_degrees(E[1])) / 2.0
    
    edge_weights = (1 / (d1 * d2)).sqrt()
    dgl_graph.edata['ew'] = edge_weights
    
    if return_adj_matrix:
        idx = torch.stack(E)
        num_nodes = dgl_graph.num_nodes()
        adj = torch.sparse_coo_tensor(
            idx, edge_weights, (num_nodes, num_nodes)
        )
        return adj
