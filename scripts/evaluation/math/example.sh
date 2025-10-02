PROMPT_TYPE="qwen-boxed"
MODEL_SAVE_NAME="qwen-2.5-math-1.5b"
save_path="./checkpoints/math"
experiment_name="qwen-2.5-math-1.5b-lr-5e-5-bz-256-max_length-3072-nproc_per_node-2-micro_batch_size-4-alpha-10"
MODEL_NAME_OR_PATH="$save_path/$experiment_name/global_step_264"

OUTPUT_DIR="./results/math/$MODEL_SAVE_NAME/$experiment_name"
n_sampling=16
temperature=1

SPLIT="test"
NUM_TEST_SAMPLE=-1

export CUDA_VISIBLE_DEVICES=0,1

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
    --use_vllm