# Friend Recommendations with Self-Rescaling Graph Neural Networks

* This is an implementation for our KDD 2022 paper: *Friend Recommendations with Self-Rescaling Graph Neural Networks*.
* Contact: xiransong@hust.edu.cn

## Dependencies

The dependencies are listed in `requirements.txt`.

## Dataset

The Pokec and LiveJournal dataset can be downloaded from here: https://drive.google.com/file/d/1MGIQyZwZQgIMn53ih6wulcrTaraxHFin/view?usp=sharing. The Xbox dataset is an industrial dataset and is not able to be public.

For a dataset (e.g. Pokec), the files included are as follows:

```
instance_Pokec
├── eval-recall
│   ├── recall_test_neg.pkl           # numpy array, size=(*, 999)
│   ├── recall_test_pos_edgelist.pkl  # numpy array, size=(*, 2)
│   ├── recall_val_neg.pkl            # numpy array, size=(*, 999)
│   └── recall_val_pos_edgelist.pkl   # numpy array, size=(*, 2)
├── eval-rank
│   ├── rank_test_neg.pkl           # numpy array, size=(*, 99)
│   ├── rank_test_pos_edgelist.pkl  # numpy array, size=(*, 2)
│   ├── rank_val_neg.pkl            # numpy array, size=(*, 99)
│   └── rank_val_pos_edgelist.pkl   # numpy array, size=(*, 2)
├── train_graph.pkl                 # dgl graph for training
└── train_undi_graph.pkl            # undirected graph
```

## Model running

The scripts to run all the models are in the `Script` directory.

To run a model, please modify the path `PROJECT_ROOT` and `ALL_DATA_ROOT` in the script, and run `bash run_xxx.sh`.
Note that prerequisite data are needed before running some models. (E.g. first run PPR to generate top-k neighbors, then run the PPRGo model.)
