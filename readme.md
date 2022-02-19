# Friend Recommendations with Self-Rescaling Graph Neural Networks

This is an implementation for our KDD 2022 paper submission.

## Dependencies

The dependencies are listed in `requirements.txt`.

## Data format

For a dataset (e.g. Pokec), the files needed are as follows:

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

To run a model (e.g. PPRGo): modify the data path `ALL_DATA_ROOT` in the script `Script/run_pprgo.sh`, check other configurations of the model, and run `bash Script/run_pprgo.sh`.
