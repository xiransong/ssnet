XBOX="3m_di_0.1"
LIVEJOURNAL="LiveJournal_0.1"
POKEC="Pokec_0.1"

# bash run_lightgcn.sh $POKEC

# bash run_lightgcn.sh $XBOX

# bash run_lightgcn.sh $LIVEJOURNAL

# bash run_pprgo.sh $POKEC

bash run_pprgo.sh $XBOX

# bash run_pprgo.sh $LIVEJOURNAL

# bash run_sage.sh $POKEC

# bash run_sage.sh $XBOX

# bash run_sage.sh $LIVEJOURNAL

# bash train_discriminator.sh $XBOX

# bash train_discriminator.sh $LIVEJOURNAL

# bash train_gan.sh $POKEC

bash train_gan.sh $XBOX

# bash train_gan.sh $LIVEJOURNAL


# bash train_gan.sh $XBOX

# bash train_discriminator.sh $POKEC

# bash train_discriminator.sh $LIVEJOURNAL

# bash generate_csr_graph.sh $POKEC

# bash run_lightgcn.sh $POKEC

# bash generate_csr_graph.sh $LIVEJOURNAL

# bash run_lightgcn.sh $LIVEJOURNAL

# bash generate_csr_graph.sh $XBOX

# bash run_lightgcn.sh $XBOX

# bash run_clustergcn.sh

# bash run_sage.sh

# bash run_gin.sh

# bash run_sagn.sh

# bash run_gat.sh

# bash run_gat.sh

#---------------------------------
# PROJECT_ROOT="/media/xreco/DEV/xiran/code/UniRec/LinkPred"
# CONFIG_FILE_ROOT=$PROJECT_ROOT"/config"

# ALL_DATA_ROOT='/media/xreco/DEV/xiran/data'

# conda activate pytorch

# #---------------------------------

# DATASET_NAME="3m_di_0.1"
# # DATASET_NAME="LiveJournal_0.1"
# # DATASET_NAME="Pokec_0.1"
# # DATASET_NAME=$1

# echo "DATASET_NAME="$DATASET_NAME

# DATA_ROOT=$ALL_DATA_ROOT'/instance_'$DATASET_NAME
# OUTPUT_ROOT=$ALL_DATA_ROOT"/gnn_"$DATASET_NAME

# #---------------------------------

# RESULTS_ROOT=$OUTPUT_ROOT"/pprgo/degree_mlp"

# python Main/new_train_normal_gnn.py $PROJECT_ROOT --data_root $DATA_ROOT --results_root $RESULTS_ROOT \
#     --model "pprgo" --config_file $CONFIG_FILE_ROOT"/common_gnn-config.yaml"  --ppr_data_root $OUTPUT_ROOT"/ppr/undirected-top100" --topk 32 \
#     --embs_lr 0.01 --from_pretrained 0 --freeze_nodes_emb 0 --file_pretrained_embs $OUTPUT_ROOT"/node2vec/0/out_emb_table.pt"  \
#     --final_layer_mlp 0 \
#     --degree_mlp 1 \
#     # --scaling 0 --l2norm 0 --ffn 0 --res_ffn 0 \
#     # --use_tao 1 \
#     # --learnable_scalar 1 \

# python Main/eval.py $PROJECT_ROOT \
#     --data_root $DATA_ROOT \
#     --model_root $RESULTS_ROOT \
#     --model_name 'base_embedding_model' \
#     --test_methods 'recall_test rank_test recall_test_whole_graph'