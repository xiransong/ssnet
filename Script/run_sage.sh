PROJECT_ROOT=".."
CONFIG_FILE_ROOT=$PROJECT_ROOT"/config"

ALL_DATA_ROOT='../../data'

# DATASET_NAME="3m_di_0.1"
# DATASET_NAME="LiveJournal_0.1"
DATASET_NAME="Pokec_0.1"

DATA_ROOT=$ALL_DATA_ROOT'/instance_'$DATASET_NAME
OUTPUT_ROOT=$ALL_DATA_ROOT"/gnn_"$DATASET_NAME

###############################################
RESULTS_ROOT=$OUTPUT_ROOT"/graphsage/normal"

python ../Main/new_train_normal_gnn.py $PROJECT_ROOT --data_root $DATA_ROOT --results_root $RESULTS_ROOT \
    --model "graphsage" --config_file $CONFIG_FILE_ROOT"/common_gnn-config.yaml" \
    --from_pretrained 1 --embs_lr 0.005 --gnn_lr 0.001 --freeze_nodes_emb 1 --file_pretrained_embs $OUTPUT_ROOT"/node2vec/0/out_emb_table.pt"  \
    --final_layer_mlp 0 --scaling 0 \
    --num_gcn_layer 1 \
    --layer_sample 1 \
    --num_layer_sample '[20, ]' \

python ../Main/eval.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --model_root $RESULTS_ROOT \
    --model_name 'base_embedding_model' \
    --test_methods 'recall_test rank_test recall_test_whole_graph' \

###############################################
RESULTS_ROOT=$OUTPUT_ROOT"/graphsage/scaling"

python ../Main/new_train_normal_gnn.py $PROJECT_ROOT --data_root $DATA_ROOT --results_root $RESULTS_ROOT \
    --model "graphsage" --config_file $CONFIG_FILE_ROOT"/common_gnn-config.yaml" \
    --from_pretrained 1 --embs_lr 0.005 --gnn_lr 0.001 --freeze_nodes_emb 1 --file_pretrained_embs $OUTPUT_ROOT"/node2vec/0/out_emb_table.pt"  \
    --final_layer_mlp 1 --scaling 1 \
    --num_gcn_layer 1 \
    --layer_sample 1 \
    --num_layer_sample '[20, ]' \

python ../Main/eval.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --model_root $RESULTS_ROOT \
    --model_name 'base_embedding_model' \
    --test_methods 'recall_test rank_test recall_test_whole_graph' \
