from Utils import io
from Utils.utils import combine_dict_list_and_calc_mean

from abc import abstractmethod
import numpy as np
import torch
from tqdm import tqdm
import os.path as osp

    

class WholeGraphTest:
    ''' test on the whole graph '''
    
    def __init__(self, model, file_pos_edgelist):
        self.model = model
        self.file_pos_edgelist = file_pos_edgelist
        
    def test(self):
        pos_edgelist = io.load_pickle(self.file_pos_edgelist)
        dl = torch.utils.data.DataLoader(dataset=pos_edgelist, batch_size=32, shuffle=False)
        
        results_list = []
        for b in tqdm(dl):
            src, dst, neg_dst = b[:,0], b[:,1], None
            batch_results = self.model.inference(
                (src, dst, neg_dst), eval_on_whole_graph=True
            )
            results_list.append(batch_results)
        test_results = combine_dict_list_and_calc_mean(results_list)
        return test_results


class SelectedNegTest:
    ''' test by 1-pos-k-neg  '''
    
    def __init__(self, model, file_pos_edgelist, file_neg_array):
        self.model = model
        self.file_pos_edgelist = file_pos_edgelist
        self.file_neg_array = file_neg_array
    
    def test(self):
        pos_edgelist = io.load_pickle(self.file_pos_edgelist)
        neg_array = io.load_pickle(self.file_neg_array)
        dl = torch.utils.data.DataLoader(
            dataset=np.concatenate((pos_edgelist, neg_array), axis=-1),
            batch_size=5120, shuffle=False)
        
        results_list = []
        for b in tqdm(dl):
            src, dst, neg_dst = b[:,0], b[:,1], b[:,2:]
            batch_results = self.model.inference(
                (src, dst, neg_dst), eval_on_whole_graph=False
            )
            results_list.append(batch_results)
        test_results = combine_dict_list_and_calc_mean(results_list)
        return test_results


class DirectionClfTest:
    ''' direction classification test '''
    
    def __init__(self, model, eval_data_root):
        self.model = model
        self.eval_data_root = eval_data_root

    def test(self):
        pos_E = io.load_pickle(osp.join(self.eval_data_root, "direction_clf_pos_edgelist.pkl"))
        neg_E = np.concatenate((pos_E[:,1].reshape(-1, 1), 
                                pos_E[:,0].reshape(-1, 1)), axis=-1)
        dl = torch.utils.data.DataLoader(
            dataset=np.concatenate((pos_E, neg_E)), 
            batch_size=5120, shuffle=False)
        
        pred_scores = []
        for b in tqdm(dl):
            src, dst, neg_dst = b[:,0], b[:,1], None
            batch_scores = self.model.inference(
                (src, dst, neg_dst), eval_on_whole_graph=False, only_return_pos_score=True
            )
            pred_scores.append(batch_scores)
        pred_scores = np.concatenate(pred_scores)
        
        accuracy = (pred_scores[:len(pos_E)] > pred_scores[-len(neg_E):]).sum() / len(pos_E)
        
        return {"accuracy": accuracy}
        

class ReversePredTest:
    ''' reverse edge prediction test '''
    
    def __init__(self, model, eval_data_root):
        self.model = model
        self.eval_data_root = eval_data_root
    
    def test(self):
        pos_E = io.load_pickle(osp.join(self.eval_data_root, "reverse_pred_pos_edgelist.pkl"))
        neg_list = io.load_pickle(osp.join(self.eval_data_root, "reverse_pred_neg.pkl"))
        dl = torch.utils.data.DataLoader(
            dataset=np.concatenate((pos_E, neg_list.reshape(-1, 1)), axis=-1), 
            batch_size=5120, shuffle=False)
        
        pred_scores = []
        for b in tqdm(dl):
            src, dst, neg_dst = b[:,0], b[:,1], b[:,2]
            batch_scores = self.model.inference(
                (src, dst, neg_dst), eval_on_whole_graph=False, only_return_pos_neg_score=True
            )
            pred_scores.append(batch_scores)
        pred_scores = np.concatenate(pred_scores)
        
        accuracy = (pred_scores[:, 0] > pred_scores[:, 1]).sum() / len(pos_E)
        
        return {"accuracy": accuracy}
