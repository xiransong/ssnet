import torch
import dgl
import numpy as np
from tqdm import tqdm


def remove_redundant_edges_in_edgelist(edgelist: np.array):
    edge_set = set()
    for e in tqdm(edgelist, desc="array to set"):
        edge_set.add((e[0], e[1]))
    new_edgelist = np.empty(shape=(len(edge_set), 2), dtype=np.int32)
    for i, e in tqdm(edge_set, desc="set to array"):
        new_edgelist[i] = np.array(e)
    return new_edgelist


def generate_2hop_neg_for_one_src(dgl_undi_graph, src, pos, num_neg=99, num_one_hop_check=30):
    nei1 = dgl_undi_graph.successors(src).numpy()
    np.random.shuffle(nei1)
    candidates = set()
    for v in nei1[:num_one_hop_check]:
        candidates.update(dgl_undi_graph.successors(v).tolist())
    nei1 = set(nei1)  # 1-hop
    nei1.add(src)
    nei1.add(pos)
    two_hop_neg = list(candidates - nei1)
    
    if len(two_hop_neg) >= num_neg:
        two_hop_neg = np.array(two_hop_neg)
        np.random.shuffle(two_hop_neg)
        return two_hop_neg[:num_neg]
    else:
        return None


def generate_2hop_neg(train_undi_graph, pos_edgelist, num_neg=99, num_one_hop_check=30):
    _pos_edgelist = []
    neg_array = []
    skip = 0
    for src, pos in tqdm(pos_edgelist):
        two_hop_neg = generate_2hop_neg_for_one_src(train_undi_graph, src, pos, num_neg, num_one_hop_check)
        if two_hop_neg is None:
            skip += 1
            continue
        _pos_edgelist.append((src, pos))
        neg_array.append(two_hop_neg)
    _pos_edgelist = np.array(_pos_edgelist)
    neg_array = np.array(neg_array)
    print("skip:", skip)
    print("neg_array:", neg_array.shape)
    return _pos_edgelist, neg_array


def generate_undirected_dgl_graph(g: dgl.DGLGraph):
    E0, E1 = g.edges()
    E0, E1 = E0.numpy(), E1.numpy()
    edge_set = set()
    for e in tqdm(zip(E0, E1), desc="array to set"):
        edge_set.add((e[0], e[1]))
        edge_set.add((e[1], e[0]))
    uE0 = np.empty(len(edge_set), dtype=E0.dtype)
    uE1 = np.empty(len(edge_set), dtype=E1.dtype)
    for i, e in tqdm(enumerate(edge_set), desc="set to array"):
        uE0[i] = e[0]
        uE1[i] = e[1]
    undi_graph = dgl.graph((uE0, uE1))
    return undi_graph


def get_edge_pairs_of_undirected_graph(graph: dgl.DGLGraph):
    """
    graph: assume to be undirected
    """
    num_edges = graph.num_edges()
    edges = graph.edges()
    src_nids = edges[0].numpy()
    dst_nids = edges[1].numpy()
    node2eid_dict = {}  # map (src, dst) to eid
    edge_pairs = []  # [ (eid, reverse_eid), ... ]
    for eid in range(num_edges):
        src = src_nids[eid]
        dst = dst_nids[eid]
        reverse_eid = node2eid_dict.get((dst, src))
        if reverse_eid is None:
            node2eid_dict[(src, dst)] = eid
        else:
            edge_pairs.append((eid, reverse_eid))
    return torch.tensor(edge_pairs, dtype=torch.int64)


def get_reverse_eid_map(edge_pairs: torch.Tensor):
    _edge_pairs = edge_pairs.numpy()
    num_total_edges = 2 * edge_pairs.size()[0]
    reverse_eid_map = torch.empty(num_total_edges, dtype=torch.int64)
    for e in _edge_pairs:
        reverse_eid_map[e[0]] = e[1]
        reverse_eid_map[e[1]] = e[0]
    return reverse_eid_map


def get_reverse_eid_map_from_graph(graph: dgl.DGLGraph):
    """
    graph: assume to be undirected
    """
    edge_pairs = get_edge_pairs_of_undirected_graph(graph)
    return get_reverse_eid_map(edge_pairs)


def neg_dst_sample(graph: dgl.DGLGraph, eids, neg_per_edge=300):
    src, dst = graph.find_edges(eids)
    neg_dst = torch.randint(0, graph.num_nodes(), (len(src), neg_per_edge))
    return src, dst, neg_dst


def make_pos_neg_pairs(src, dst, neg_dst):

    pos_pairs = torch.tensor((src.numpy(), dst.numpy())).t()

    _src = src.repeat_interleave(neg_dst.shape[-1])
    _neg_dst = neg_dst.flatten()

    neg_pairs = torch.tensor((_src.numpy(), _neg_dst.numpy())).t()

    return pos_pairs, neg_pairs



def trim_graph_by_min_out_degree(graph: dgl.DGLGraph, min_out_degree):
    while True:
        degrees = graph.out_degrees()
        nodes_to_be_removed = graph.nodes()[degrees < min_out_degree]
        if len(nodes_to_be_removed) == 0:
            break
        graph.remove_nodes(nodes_to_be_removed)
