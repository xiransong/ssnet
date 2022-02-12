PROJECT_ROOT="/media/xreco/DEV/xiran/code/UniRec/LinkPred"
ALL_DATA_ROOT='/media/xreco/DEV/xiran/data'

DATASET_NAME=$1
DATA_ROOT=$ALL_DATA_ROOT'/instance_'$DATASET_NAME

python Main/generate_csr_graph.py $PROJECT_ROOT --data_root $DATA_ROOT
