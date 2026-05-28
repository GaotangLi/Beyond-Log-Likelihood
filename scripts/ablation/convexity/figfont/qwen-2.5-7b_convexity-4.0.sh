#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

nproc_per_node=2
project_name=figlet-font
lr=5e-5
bz=256
max_length=800
micro_batch_size=16
weight_decay=1e-5

experiment_name=qwen-2.5-7b-lr-$lr-bz-$bz-max_length-$max_length-nproc_per_node-$nproc_per_node-micro_batch_size-$micro_batch_size-weight_decay-$weight_decay-0916-convexity-4.0
save_path=./checkpoints/ablation/convexity_figfont/$experiment_name

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
        -m main_verl.trainer.fsdp_sft_trainer \
    data.train_files=./data/figfont/train.parquet \
    data.val_files=./data/figfont/val.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.train_batch_size=$bz \
    data.max_length=$max_length \
    optim.lr=$lr \
    data.prompt_dict_keys=['qwen_prompt'] \
    data.response_dict_keys=['answer'] \
    data.use_original_prompt=True \
    data.micro_batch_size_per_gpu=$micro_batch_size \
    model.partial_pretrain=Qwen/Qwen2.5-7B \
    model.use_liger=True \
    model.fsdp_config.model_dtype=bf16 \
    optim.weight_decay=$weight_decay \
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
    trainer.objective_trans=convexity-4.0

MODEL_SAVE_NAME="qwen-2.5-7b"
MODEL_NAME_OR_PATH="$save_path/global_step_156"
export CUDA_VISIBLE_DEVICES

python evaluations/figfont/main.py \
    --model_name ${MODEL_NAME_OR_PATH} \
    --output_file_name ${experiment_name} \
    --model_save_name ${MODEL_SAVE_NAME} \
    --tensor_parallel_size ${nproc_per_node}
