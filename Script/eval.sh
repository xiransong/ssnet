PROJECT_ROOT="/media/xreco/DEV/xiran/code/UniRec/LinkPred"

conda activate pytorch

ALL_DATA_ROOT='/media/xreco/DEV/xiran/data'

# DATASET_NAME="3m_di_0.1"
# DATASET_NAME="LiveJournal_0.1"
DATASET_NAME="Pokec_0.1"

DATA_ROOT=$ALL_DATA_ROOT'/instance_'$DATASET_NAME
OUTPUT_ROOT=$ALL_DATA_ROOT"/gnn_"$DATASET_NAME
RESULTS_ROOT=$OUTPUT_ROOT'/random'

python Main/eval.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --model_root $RESULTS_ROOT \
    --model_name 'base_embedding_model' \
    --test_methods 'recall_test recall_test_whole_graph rank_test'


DATASET_NAME="3m_di_0.1"
# DATASET_NAME="LiveJournal_0.1"
# DATASET_NAME="Pokec_0.1"

DATA_ROOT=$ALL_DATA_ROOT'/instance_'$DATASET_NAME
OUTPUT_ROOT=$ALL_DATA_ROOT"/gnn_"$DATASET_NAME
RESULTS_ROOT=$OUTPUT_ROOT'/random'

python Main/eval.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --model_root $RESULTS_ROOT \
    --model_name 'base_embedding_model' \
    --test_methods 'recall_test recall_test_whole_graph rank_test'


# DATASET_NAME="3m_di_0.1"
DATASET_NAME="LiveJournal_0.1"
# DATASET_NAME="Pokec_0.1"

DATA_ROOT=$ALL_DATA_ROOT'/instance_'$DATASET_NAME
OUTPUT_ROOT=$ALL_DATA_ROOT"/gnn_"$DATASET_NAME
RESULTS_ROOT=$OUTPUT_ROOT'/random'

python Main/eval.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --model_root $RESULTS_ROOT \
    --model_name 'base_embedding_model' \
    --test_methods 'recall_test recall_test_whole_graph rank_test'