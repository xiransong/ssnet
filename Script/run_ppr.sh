PROJECT_ROOT=".."
CONFIG_FILE_ROOT=$PROJECT_ROOT"/config"

ALL_DATA_ROOT='../../data'

# DATASET_NAME="3m_di_0.1"
# DATASET_NAME="LiveJournal_0.1"
DATASET_NAME="Pokec_0.1"

DATA_ROOT=$ALL_DATA_ROOT'/instance_'$DATASET_NAME
OUTPUT_ROOT=$ALL_DATA_ROOT"/gnn_"$DATASET_NAME

RESULTS_ROOT=$OUTPUT_ROOT"/ppr/undirected-top100"

python ../Main/run_ppr.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --results_root $RESULTS_ROOT \
    --num_walks 1000 \
    --walk_length 30 \
    --alpha 0.7 \
    --topk 100
