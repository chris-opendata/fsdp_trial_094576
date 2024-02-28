#!/bin/bash


help()
{
    echo ". run.training.mgpu.sh"
    echo "      --dataset_name                [cnn|xsum]"
    echo "      --dataset_config_name         [3.0.0|none]"
    echo "      --pretrained_model            [facebook/bart-large-cnn|facebook/bart-large-xsum]"
    echo "      --gpu_ids                     [0,1,2]"
    echo "      --process_port_id             [9999]"
    echo "      --bilinear_dim                [1024]"
    echo "      --n_train_data_samples        [100|-1]"
    echo "      --per_device_train_batch_size [4]"
    echo "      --dataset_cache_dir           [directory where datasets are downloaded|none]"
#    echo "      --pretrained_model_cache_dir  [directory where pretrained model weights and tokenizer vocabulary are downloaded|none]"
    echo "      --pyvenv_dir                  [Python virtual env directory (parent to bin)]"
    echo "Example:"
    echo ". run.training.mgpu.sh --dataset_name cnn_dailymail --dataset_config_name 3.0.0 --pretrained_model facebook/bart-large-cnn --gpu_ids 0,1 --process_port_id 9999 --bilinear_dim 1024 --n_train_data_samples 100 --per_device_train_batch_size 4 --pyvenv_dir ~/dev/pyvenv_pt2 --dataset_cache_dir ~/Data/dev/.cache/huggingface/datasets"
}

NUM_ARGUMENTS=$#
EXPECTED_N_ARGS=20
if [ "$NUM_ARGUMENTS" -ne ${EXPECTED_N_ARGS} ]; then
    help
    return
fi

while :
do
  case "$1" in
    --dataset_name )
      DATASET_NAME="$2"
      shift 2
      ;;
    --dataset_config_name )
      DATASET_CONFIG_NAME="$2"
      shift 2
      ;;
    --pretrained_model )
      PRETRAINED_MODEL_TYPE="$2"
      shift 2
      ;;
    --gpu_ids )
      GPU_IDS="$2"
      shift 2
      ;;
    --process_port_id )
      PROCESS_PORT_ID="$2"
      shift 2
      ;;
    --bilinear_dim )
      BILINEAR_DIM="$2"
      shift 2
      ;;
    --n_train_data_samples )
      N_TRAIN_DATA_SAMPLES="$2"
      shift 2
      ;;
    --per_device_train_batch_size )
      PER_DEVICE_TRAIN_BATCH_SIZE="$2"
      shift 2
      ;;
    --dataset_cache_dir )
      DATASET_CACHE_DIR="$2"
      shift 2
      ;;
    --pretrained_model_cache_dir )
      PRETRAINED_MODEL_CACHE_DIR="$2"
      shift 2
      ;;
    --pyvenv_dir )
      PYVENV_DIR="$2"
      shift 2
      ;;
    --)
      shift;
      break
      ;;
    *)
      # echo "Unexpected option: $1"
      # help
      break
      ;;
  esac
done


source ${PYVENV_DIR}/bin/activate
export PYTHONPATH="$PYTHONPATH:$PWD:$PWD/.."
# export PATH=/usr/local/cuda-11.3/bin:$PATH
# export CPATH=/usr/local/cuda-11.3/include:$CPATH
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.3/lib64
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO


FOLDER_NAME="`basename $PWD`"

RUN_LOG_DIR="./runlog"
[ -d ${RUN_LOG_DIR} ] || mkdir -p ${RUN_LOG_DIR}

today=`date '+%Y_%m_%d_%H_%M'`;
RUN_LOG="${RUN_LOG_DIR}/train_log_$today.out"

echo ${RUN_LOG}
echo $HOSTNAME >${RUN_LOG}

echo "--dataset_name:                 ${DATASET_NAME}"
echo "--dataset_config_name           ${DATASET_CONFIG_NAME}"
echo "--base_model_pretrained_name:   ${PRETRAINED_MODEL_TYPE}"
echo "--tokenizer_name:               ${PRETRAINED_MODEL_TYPE}"
echo "--gpu_ids:                      ${GPU_IDS}"
echo "--process_port_id:              ${PROCESS_PORT_ID}"
echo "--bilinear_dim:                 ${BILINEAR_DIM}"
echo "--per_device_train_batch_size:  ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "--n_train_data_samples:         ${N_TRAIN_DATA_SAMPLES}"
echo "--dataset_cache_dir:            ${DATASET_CACHE_DIR}"
#echo "--pretrained_model_cache_dir    ${PRETRAINED_MODEL_CACHE_DIR}"

PARAMS="--dataset_name ${DATASET_NAME}  "
PARAMS=${PARAMS}"--dataset_config_name ${DATASET_CONFIG_NAME} "
PARAMS=${PARAMS}"--base_model_pretrained_name ${PRETRAINED_MODEL_TYPE}  "
PARAMS=${PARAMS}"--tokenizer_name ${PRETRAINED_MODEL_TYPE}  "
PARAMS=${PARAMS}"--use_slow_tokenizer  "
PARAMS=${PARAMS}"--seed 25687394  "
PARAMS=${PARAMS}"--bilinear_dim ${BILINEAR_DIM} "
PARAMS=${PARAMS}"--per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} "
PARAMS=${PARAMS}"--pad_to_max_length "
PARAMS=${PARAMS}"--n_train_data_samples ${N_TRAIN_DATA_SAMPLES} "
PARAMS=${PARAMS}"--dataset_cache_dir ${DATASET_CACHE_DIR} "
#PARAMS=${PARAMS}"--pretrained_model_cache_dir ${PRETRAINED_MODEL_CACHE_DIR} "

MAIN_APP=train_main.py

GPU_ID_LIST=(${GPU_IDS//,/ })
NUM_GPUS=${#GPU_ID_LIST[@]}
echo "num_gpus:    ${NUM_GPUS}"

if [ "${NUM_GPUS}" -gt "1" ]; then
    echo "CUDA_VISIBLE_DEVICES=${GPU_IDS} accelerate launch --config_file ./accelerate_config.yaml ${MAIN_APP} ${PARAMS}"
    CUDA_VISIBLE_DEVICES=${GPU_IDS} accelerate launch --config_file ./accelerate_config.yaml ${MAIN_APP} ${PARAMS} >>${RUN_LOG} 2>&1 &
else
    echo "CUDA_VISIBLE_DEVICES=${GPU_IDS} accelerate launch --main_process_port=${PROCESS_PORT_ID} --mixed_precision=no --num_processes=1 --num_machines=1 ${MAIN_APP} ${PARAMS}"
    CUDA_VISIBLE_DEVICES=${GPU_IDS} accelerate launch --main_process_port=${PROCESS_PORT_ID} --mixed_precision=no --num_processes=1 --num_machines=1 ${MAIN_APP} ${PARAMS} >>${RUN_LOG} 2>&1 &
fi
