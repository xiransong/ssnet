import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from Utils import io
from Utils.parse_arguments import parse_arguments
from Utils.utils import print_dict, ensure_dir
from Wrapper.dataloader import new_build_val_test_dataloader
import Module.gnn as gnn_module
from Model.FastNode2Vec import FastNode2Vec
from Trainer.TrainTracer import TrainTracer

import torch
import numpy as np
import gensim
import os.path as osp
import time
import setproctitle


class ConvergeException(Exception):
    pass


class N2VCallBack(gensim.models.callbacks.CallbackAny2Vec):
    
    def __init__(self, num_nodes, results_root, fn_eval, convergence_threshold):
        self.epoch = 0
        self.num_nodes = num_nodes
        self.results_root = results_root
        self.fn_eval = fn_eval
        self.train_tracer = TrainTracer(convergence_threshold=convergence_threshold, 
                                        record_root=results_root)
        self.idx = np.arange(self.num_nodes).tolist()

    def on_epoch_end(self, model):
        self.epoch += 1
        
        embs = torch.FloatTensor(model.wv[self.idx])
        
        print("val...")
        val_results = self.fn_eval(embs)
        print(val_results)
        
        key_score = val_results['ndcg']
        
        def save_best_model():
            io.save_pickle(osp.join(self.results_root, "n2v-embeddings.pkl"), embs)
        
        is_converged = self.train_tracer.check_and_save(key_score, self.epoch, val_results, save_best_model)
        if is_converged:
            raise ConvergeException


def main():
    
    parsed_results = parse_arguments()
    '''
    cmd arg requirements:
    --data_root
    --results_root
    --config_file
    
    '''
    
    config_file = parsed_results['config_file']
    config = io.load_yaml(config_file)
    config.update(parsed_results)
    print_dict(config)
    
    data_root = config['data_root']
    results_root = config['results_root']
    ensure_dir(results_root)
    io.save_yaml(osp.join(results_root, "config.yaml"), config)
    
    train_graph = io.load_pickle(osp.join(data_root, "train_graph.pkl"))
    
    val_dl, test_dl = new_build_val_test_dataloader({}, config)
    
    def eval_embs_on_val_set(embs):
        val_results = gnn_module.eval_by_out_emb_table(embs, val_dl)
        return val_results
    
    model = FastNode2Vec(train_graph.num_nodes(), 
                         train_graph.edges()[0].numpy(), 
                         train_graph.edges()[1].numpy())

    callback = N2VCallBack(train_graph.num_nodes(), 
                           results_root, 
                           eval_embs_on_val_set,
                           convergence_threshold=config['convergence_threshold'])
    
    # generate the same sentences for each epoch (for different sentences, set walk_seed=None)
    walk_seed = None
    
    try:
        model.run_node2vec(
            dim=config['dim'], 
            epochs=config['epochs'], 
            alpha_schedule=config['alpha_schedule'],
            num_walks=config['num_walks'], 
            walk_length=config['walk_length'], 
            window=config['window'], 
            p=config['p'], 
            q=config['q'],
            walk_seed=walk_seed,
            callbacks=[callback]
        )
    except (ConvergeException, KeyboardInterrupt):
        pass
    
    print("test...")
    embs = io.load_pickle(osp.join(results_root, "n2v-embeddings.pkl"))
    torch.save(embs, osp.join(results_root, "out_emb_table.pt"))
    test_results = gnn_module.eval_by_out_emb_table(embs, test_dl)
    print(test_results)
    io.save_json(osp.join(results_root, "n2v-test_results.json"), test_results)


if __name__ == "__main__":
    
    setproctitle.setproctitle('n2v-' + 
                              time.strftime("%m%d-%H%M%S", time.localtime(time.time())))
    
    main()
