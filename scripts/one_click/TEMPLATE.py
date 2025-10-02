MODEL_MAPPING = {
    "qwen-2.5-math-1.5b": "Qwen/Qwen2.5-Math-1.5B",
    "qwen-2.5-math-7b": "Qwen/Qwen2.5-Math-7B",
    "qwen-2.5-1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    "deepseek-math-7b": "deepseek-ai/deepseek-math-7b-base",
}


HYPERPARAMETER_MAPPING = {
    "math": {
        "qwen-2.5-math-1.5b": {
            "lr": "5e-5",
            "case": "qwen-boxed"
        },
        "qwen-2.5-math-7b": {
            "lr": "5e-5",
            "case": "qwen-boxed"
        },
        "llama-3.1-8b": {
            "lr": "2e-5",
            "case": "llama-base-boxed-with-bos"
        },
        "deepseek-math-7b": {
            "lr": "5e-5",
            "case": "deepseek-math"
        }
    },
    "medical": {
        "llama-3.1-3b": {
            "lr": "5e-5",
            "prompt_dict_key": "llama_prompt",
            "response_dict_key": "llama_reponse_no_thinking"
        },
        "llama-3.1-8b": {
            "lr": "2e-5",
            "prompt_dict_key": "llama_prompt",
            "response_dict_key": "llama_reponse_no_thinking"
        },
        "qwen-2.5-1.5b": {
            "lr": "5e-5",
            "prompt_dict_key": "qwen_prompt",
            "response_dict_key": "qwen_reponse_no_thinking"
        },
        "qwen-2.5-math-7b": {
            "lr": "5e-5",
            "prompt_dict_key": "qwen_prompt",
            "response_dict_key": "qwen_reponse_no_thinking"
        }
    },
    "figfont": {
        "qwen-2.5-7b": {
            "lr": "5e-5",
            "prompt_dict_key": "qwen_prompt",
            "response_dict_key": "answer"
        },
        "qwen-2.5-1.5b": {
            "lr": "5e-5",
            "prompt_dict_key": "qwen_prompt",
            "response_dict_key": "answer"
        },
        "llama-3.1-8b": {
            "lr": "2e-5",
            "prompt_dict_key": "llama_prompt",
            "response_dict_key": "answer"
        },
        "llama-3.2-3b": {
            "lr": "5e-5",
            "prompt_dict_key": "llama_prompt",
            "response_dict_key": "answer"
        },
    }
}


MATH_SCRIPT = r"""nproc_per_node={nproc_per_node}
project_name=math
lr={lr}
bz=256
max_length=3072
micro_batch_size=4

experiment_name={model_save_name}-lr-$lr-bz-$bz-max_length-$max_length-nproc_per_node-$nproc_per_node-micro_batch_size-$micro_batch_size-{trainer_objective_trans}
save_path=./checkpoints/math/$experiment_name

CUDA_VISIBLE_DEVICES={cuda_visible_devices} torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
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
    model.partial_pretrain={base_model_official_path} \
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
    trainer.objective_trans={trainer_objective_trans}


PROMPT_TYPE="{case}"
MODEL_NAME_OR_PATH="$save_path/global_step_264"
OUTPUT_DIR="./results/math/{model_save_name}/$experiment_name"  
n_sampling=16
temperature=1

SPLIT="test"
NUM_TEST_SAMPLE=-1

export CUDA_VISIBLE_DEVICES={cuda_visible_devices}

DATA_NAME="minerva_math,olympiadbench,aime24,amc23,math_oai"

TOKENIZERS_PARALLELISM=false \
python3 -u evaluations/math/math_eval_multigpu.py \
    --model_name_or_path ${{MODEL_NAME_OR_PATH}} \
    --data_name ${{DATA_NAME}} \
    --output_dir ${{OUTPUT_DIR}} \
    --split ${{SPLIT}} \
    --prompt_type ${{PROMPT_TYPE}} \
    --num_test_sample ${{NUM_TEST_SAMPLE}} \
    --seed 0 \
    --temperature ${{temperature}} \
    --n_sampling ${{n_sampling}} \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --tensor_parallel_size ${{nproc_per_node}} \
    --gpu_memory_utilization 0.95"""







MEDICAL_SCRIPT = r"""nproc_per_node={nproc_per_node}
project_name=medical
lr={lr}
bz=256
max_length=800
micro_batch_size=16

experiment_name={model_save_name}-lr-$lr-bz-$bz-max_length-$max_length-nproc_per_node-$nproc_per_node-micro_batch_size-$micro_batch_size-{trainer_objective_trans}
save_path=./checkpoints/medical/$experiment_name

CUDA_VISIBLE_DEVICES={cuda_visible_devices} torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
        -m main_verl.trainer.fsdp_sft_trainer \
    data.train_files=./data/medical/train.parquet \
    data.val_files=./data/medical/val.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.train_batch_size=$bz \
    data.max_length=$max_length \
    optim.lr=$lr \
    data.prompt_dict_keys=['{prompt_dict_key}'] \
    data.response_dict_keys=['{response_dict_key}'] \
    data.use_original_prompt=True \
    data.micro_batch_size_per_gpu=$micro_batch_size \
    model.partial_pretrain={base_model_official_path} \
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
    trainer.objective_trans={trainer_objective_trans}


MODEL_NAME_OR_PATH="$save_path/global_step_91"
OUTPUT_DIR="./results/medical/{model_save_name}/$experiment_name"  

export CUDA_VISIBLE_DEVICES={cuda_visible_devices}
python evaluations/medical/main.py --model_name ${{MODEL_NAME_OR_PATH}} --output_file_name ${{experiment_name}} --model_save_name {model_save_name} --tensor_parallel_size {nproc_per_node} --max_tokens 8000"""


FIGFONT_SCRIPT = r"""nproc_per_node={nproc_per_node}
project_name=figfont
lr={lr}
bz=256
max_length=800
micro_batch_size=16
weight_decay=1e-5 

experiment_name={model_save_name}-lr-$lr-bz-$bz-max_length-$max_length-nproc_per_node-$nproc_per_node-micro_batch_size-$micro_batch_size-weight_decay-$weight_decay-{trainer_objective_trans}
save_path=./checkpoints/figfont/$experiment_name

CUDA_VISIBLE_DEVICES={cuda_visible_devices} torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
        -m main_verl.trainer.fsdp_sft_trainer \
    data.train_files=./data/figfont/train.parquet \
    data.val_files=./data/figfont/val.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.train_batch_size=$bz \
    data.max_length=$max_length \
    optim.lr=$lr \
    data.prompt_dict_keys=['{prompt_dict_key}'] \
    data.response_dict_keys=['{response_dict_key}'] \
    data.use_original_prompt=True \
    data.micro_batch_size_per_gpu=$micro_batch_size \
    model.partial_pretrain={base_model_official_path} \
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
    trainer.objective_trans={trainer_objective_trans}


MODEL_NAME_OR_PATH="$save_path/global_step_156"
OUTPUT_DIR="./results/figfont/{model_save_name}/$experiment_name"  

export CUDA_VISIBLE_DEVICES={cuda_visible_devices}
python evaluations/figfont/main.py --model_name ${{MODEL_NAME_OR_PATH}} --output_file_name ${{experiment_name}} --model_save_name {model_save_name} --tensor_parallel_size {nproc_per_node}"""