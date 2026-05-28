#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

nproc_per_node=2
project_name=numina-cot
lr=5e-5
bz=256
max_length=3072
micro_batch_size=8

experiment_name=numina-cot-qwen-2.5-math-1.5b-lr-$lr-bz-$bz-max_length-$max_length-nproc_per_node-$nproc_per_node-micro_batch_size-$micro_batch_size-figure5-UnlikehoodTop-K-60
save_path=./checkpoints/ablation/figure5/unlikehood_top/$experiment_name

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
        -m main_verl.trainer.fsdp_sft_trainer \
    data.train_files=./data/math/train.parquet \
    data.val_files=./data/math/val.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.train_batch_size=$bz \
    data.max_length=$max_length \
    optim.lr=$lr \
    data.prompt_dict_keys=['question'] \
    data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=$micro_batch_size \
    model.partial_pretrain=Qwen/Qwen2.5-Math-1.5B \
    model.use_liger=True \
    model.fsdp_config.model_dtype=bf16 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$project_name \
    trainer.experiment_name="$experiment_name-$(date +%Y%m%d-%H%M%S)" \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null \
    trainer.test_freq=100000 \
    trainer.save_freq=264 \
    trainer.total_epochs=1 \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true \
    trainer.objective_trans=UnlikehoodTop-0.9985600709915161

PROMPT_TYPE="qwen-boxed"
MODEL_SAVE_NAME="qwen-2.5-math-1.5b"
MODEL_NAME_OR_PATH="$save_path/global_step_264"
OUTPUT_DIR="./results/ablation/figure5/unlikehood_top/$MODEL_SAVE_NAME/$experiment_name"
n_sampling=16
temperature=1

SPLIT="test"
NUM_TEST_SAMPLE=-1

export CUDA_VISIBLE_DEVICES

DATA_NAME="minerva_math,olympiadbench,aime24,amc23,math_oai"

TOKENIZERS_PARALLELISM=false \
python3 -u evaluations/math/math_eval_multigpu.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature ${temperature} \
    --n_sampling ${n_sampling} \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --tensor_parallel_size ${nproc_per_node} \
    --gpu_memory_utilization 0.95
