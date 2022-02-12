import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from Utils import io
from Utils.parse_arguments import parse_arguments
from Utils.utils import print_dict
from Main.run_ppr import get_np_csr_graph

import os.path as osp


def main():
    
    config = parse_arguments()
    print_dict(config)
    
    data_root = config['data_root']
    
    get_np_csr_graph(
        data_root, 
        csr_file=osp.join(data_root, "train_graph.np_csr.pkl"), 
        dgl_graph_file=osp.join(data_root, "train_graph.pkl")
    )


if __name__ == "__main__":
    
    main()
