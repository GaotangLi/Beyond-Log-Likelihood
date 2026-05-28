experiment_name="qwen2.5-7b-lr-5e-6-bz-256-max_length-2048-nproc_per_node-2-weight_decay-1e-4-micro_batch_size-8-original"
MODEL_NAME_OR_PATH="./checkpoints/instruction_tuning/$experiment_name/global_step_545"

python evaluations/instruction_tuning/main.py \
    --model_path ${MODEL_NAME_OR_PATH} \
    --model_name ${experiment_name} \
    --tensor_parallel_size 2
