save_path="./checkpoints/coder"
experiment_name="qwen2.5-coder-7b-lr-5e-5-bz-256-max_length-3096-nproc_per_node-2-micro_batch_size-4-original"
MODEL_NAME_OR_PATH="$save_path/$experiment_name/global_step_293"

export CUDA_VISIBLE_DEVICES=0,1
python evaluations/coder/main.py \
    --model ${MODEL_NAME_OR_PATH} \
    --tp 2
