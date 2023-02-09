PROJECT_ROOT=".."
CONFIG_FILE_ROOT=$PROJECT_ROOT"/config"

ALL_DATA_ROOT='../../data'

# DATASET_NAME="3m_di_0.1"
# DATASET_NAME="LiveJournal_0.1"
DATASET_NAME="Pokec_0.1"

DATA_ROOT=$ALL_DATA_ROOT'/instance_'$DATASET_NAME
OUTPUT_ROOT=$ALL_DATA_ROOT"/gnn_"$DATASET_NAME

##################################################
RESULTS_ROOT=$OUTPUT_ROOT"/lightgcn/normal"

python ../Main/new_train_normal_gnn.py $PROJECT_ROOT --data_root $DATA_ROOT --results_root $RESULTS_ROOT \
    --model "lightgcn" --config_file $CONFIG_FILE_ROOT"/common_gnn-config.yaml" \
    --embs_lr 0.02 --from_pretrained 0 --freeze_nodes_emb 0 --file_pretrained_embs $OUTPUT_ROOT"/lightgcn/1layer-0/normal/out_emb_table.pt"  \
    --final_layer_mlp 0  --scaling 0 \

python ../Main/eval.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --model_root $RESULTS_ROOT \
    --model_name 'base_embedding_model' \
    --test_methods 'recall_test recall_test_whole_graph rank_test'

##################################################
RESULTS_ROOT=$OUTPUT_ROOT"/lightgcn/scaling"

python ../Main/new_train_normal_gnn.py $PROJECT_ROOT --data_root $DATA_ROOT --results_root $RESULTS_ROOT \
    --model "lightgcn" --config_file $CONFIG_FILE_ROOT"/common_gnn-config.yaml" \
    --embs_lr 0.02 --from_pretrained 0 --freeze_nodes_emb 0 --file_pretrained_embs $OUTPUT_ROOT"/lightgcn/1layer-0/normal/out_emb_table.pt"  \
    --final_layer_mlp 1  --scaling 1 \

python ../Main/eval.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --model_root $RESULTS_ROOT \
    --model_name 'base_embedding_model' \
    --test_methods 'recall_test recall_test_whole_graph rank_test'
