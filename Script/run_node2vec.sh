PROJECT_ROOT=".."
CONFIG_FILE_ROOT=$PROJECT_ROOT"/config"

ALL_DATA_ROOT='../../data'

# DATASET_NAME="3m_di_0.1"
# DATASET_NAME="LiveJournal_0.1"
DATASET_NAME="Pokec_0.1"

DATA_ROOT=$ALL_DATA_ROOT'/instance_'$DATASET_NAME
OUTPUT_ROOT=$ALL_DATA_ROOT"/gnn_"$DATASET_NAME

RESULTS_ROOT=$OUTPUT_ROOT"/node2vec/0"

python ../Main/new_node2vec.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --results_root $RESULTS_ROOT \
    --config_file $CONFIG_FILE_ROOT"/n2v-config.yaml" \
    --num_walks 20 --walk_length 20 --window 3 --p 1.0 --q 10.0 \

python ../Main/eval.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --model_root $RESULTS_ROOT \
    --model_name 'base_embedding_model' \
    --test_methods 'recall_test recall_test_whole_graph rank_test'
