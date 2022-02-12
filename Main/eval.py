import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from Utils import io
from Utils.parse_arguments import parse_arguments
from Utils.utils import get_formatted_results
from Refactor.model import BaseEmbeddingModel
from Wrapper.eval import WholeGraphTest, SelectedNegTest, DirectionClfTest

import setproctitle
import os.path as osp


def main():
    
    config = parse_arguments()
    
    data_root = config['data_root']
    model_root = config['model_root']
    model_name = config['model_name']
    test_methods = config['test_methods'].split()
    ''' 
        available test methods:
        
        recall_test (1-pos-999-neg)
        recall_test_whole_graph
        rank_test
        direction_clf
        far_pos (1-pos-999-neg)
        far_pos_whole_graph
    '''
   
    model = {
        "base_embedding_model": BaseEmbeddingModel,
    }[model_name]()
    
    model.load(model_root)
    
    for test in test_methods:
        print("test:", test)
        if test == 'recall_test':
            tester = SelectedNegTest(
                model,
                file_pos_edgelist=osp.join(data_root, "eval-recall", "recall_test_pos_edgelist.pkl"),
                file_neg_array=osp.join(data_root, "eval-recall", "recall_test_neg.pkl")
            )
        elif test == 'recall_test_whole_graph':
            tester = WholeGraphTest(
                model,
                file_pos_edgelist=osp.join(data_root, "eval-recall", "recall_test_pos_edgelist.pkl")
            )
        elif test == 'rank_test':
            tester = SelectedNegTest(
                model,
                file_pos_edgelist=osp.join(data_root, "eval-rank", "rank_test_pos_edgelist.pkl"),
                file_neg_array=osp.join(data_root, "eval-rank", "rank_test_neg.pkl")
            )
        elif test == 'direction_clf':
            tester = DirectionClfTest(
                model, 
                eval_data_root=osp.join(data_root, "eval-direction_clf")
            )
        elif test == 'far_pos':
            tester = SelectedNegTest(
                model,
                file_pos_edgelist=osp.join(data_root, "eval-far_pos", "far_pos_test_pos_edgelist.pkl"),
                file_neg_array=osp.join(data_root, "eval-far_pos", "far_pos_test_neg.pkl")
            )
        elif test == 'far_pos_whole_graph':
            tester = WholeGraphTest(model,
                                    file_pos_edgelist=osp.join(data_root, "eval-far_pos", "far_pos_test_pos_edgelist.pkl"))

        results = tester.test()
        results['formatted'] = get_formatted_results(results)
        print(test, results)
        io.save_json(osp.join(model_root, "test_results-" + test + ".json"), results)


if __name__ == "__main__":
    
    setproctitle.setproctitle("eval")
    
    main()
