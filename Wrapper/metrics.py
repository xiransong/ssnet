from Utils.metric import all_metrics
from Utils.utils import combine_dict_list_and_calc_mean

import numpy as np


class MetricsWrapper:
    
    def __init__(self):
        pass
    
    def build(self, data, config):
        self.config = config
    
    def __call__(self, out_data):
        if isinstance(out_data[0], dict):
            results_list = out_data
            results = combine_dict_list_and_calc_mean(results_list)
            return None, results['ndcg'], results
        else:
            S = np.concatenate(out_data)
            results = all_metrics(S)
            key_score = results['ndcg']
            return S, key_score, results
