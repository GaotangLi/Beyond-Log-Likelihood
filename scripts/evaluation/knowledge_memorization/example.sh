experiment_name="openbookqa-qwen2.5-14b-lr-5e-5-bz-256-max_length-600-nproc_per_node-2-micro_batch_size-8-total_epochs-1-weight_decay-1e-5-oneAnswer-noise-0.3-original"
MODEL_NAME_OR_PATH="./checkpoints/knowledge_memorization/$experiment_name/global_step_21"

python evaluations/knowledge_memorization/main.py \
    --model_name ${MODEL_NAME_OR_PATH} \
    --output_file_name ${experiment_name} \
    --model_save_name qwen2.5-14b \
    --tensor_parallel_size 2 \
    --max_tokens 128
