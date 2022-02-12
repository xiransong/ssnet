import Module.dataloader
from Utils import io

import os.path as osp


def new_build_val_test_dataloader(data, config):
    data_root = config['data_root']
    if 'recall_val_test' not in config or config['recall_val_test']:
        print("## recall_val_test")
        pos_E = io.load_pickle(osp.join(data_root, "eval-recall", "recall_val_pos_edgelist.pkl"))
        eval_on_whole_graph = 'eval_on_whole_graph' in config and config['eval_on_whole_graph']
        if not eval_on_whole_graph:
            neg_array = io.load_pickle(osp.join(data_root, "eval-recall", "recall_val_neg.pkl"))
            print("val neg_array:", neg_array.shape)
            val_dl = Module.dataloader.ValTestDataloader(
                pos_E[:,0], pos_E[:,1], neg_array, batch_size=5120
            )
        else:
            print("val pos_edgelist:", pos_E.shape)
            val_dl = Module.dataloader.ValTestDataloader(
                pos_E[:,0], pos_E[:,1], None, batch_size=32
            )
            
        pos_E = io.load_pickle(osp.join(data_root, "eval-recall", "recall_test_pos_edgelist.pkl"))
        if not eval_on_whole_graph:
            neg_array = io.load_pickle(osp.join(data_root, "eval-recall", "recall_test_neg.pkl"))
            print("test neg_array:", neg_array.shape)
            test_dl = Module.dataloader.ValTestDataloader(
                pos_E[:,0], pos_E[:,1], neg_array, batch_size=5120
            )
        else:
            print("test pos_edgelist:", pos_E.shape)
            test_dl = Module.dataloader.ValTestDataloader(
                pos_E[:,0], pos_E[:,1], None, batch_size=32
            )
    else:
        print("## rank_val_test")
        pos_E = io.load_pickle(osp.join(data_root, "eval-rank", "rank_val_pos_edgelist.pkl"))
        neg_array = io.load_pickle(osp.join(data_root, "eval-rank", "rank_val_neg.pkl"))
        print("val neg_array:", neg_array.shape)
        val_dl = Module.dataloader.ValTestDataloader(
            pos_E[:,0], pos_E[:,1], neg_array, batch_size=5120
        )
        
        pos_E = io.load_pickle(osp.join(data_root, "eval-rank", "rank_test_pos_edgelist.pkl"))
        neg_array = io.load_pickle(osp.join(data_root, "eval-rank", "rank_test_neg.pkl"))
        print("test neg_array:", neg_array.shape)
        test_dl = Module.dataloader.ValTestDataloader(
            pos_E[:,0], pos_E[:,1], neg_array, batch_size=5120
        )
    return val_dl, test_dl
