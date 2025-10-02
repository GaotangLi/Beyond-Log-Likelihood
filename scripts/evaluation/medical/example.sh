save_path="./checkpoints/medical"
experiment_name="qwen-2.5-math-7b-lr-5e-5-bz-256-max_length-800-nproc_per_node-2-micro_batch_size-16-logp"
MODEL_NAME_OR_PATH="$save_path/$experiment_name/global_step_91"

export CUDA_VISIBLE_DEVICES=0,1
python evaluations/medical/main.py --model_name ${MODEL_NAME_OR_PATH} --output_file_name ${experiment_name} --model_save_name qwen-2.5-math-7b --tensor_parallel_size 2 --max_tokens 8000