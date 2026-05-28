nproc_per_node=2
project_name=coder-sft
lr=5e-5
bz=256
max_length=3096
micro_batch_size=4

experiment_name=qwen2.5-coder-7b-lr-$lr-bz-$bz-max_length-$max_length-nproc_per_node-$nproc_per_node-micro_batch_size-$micro_batch_size-logp
save_path=./checkpoints/coder/$experiment_name

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
        -m main_verl.trainer.fsdp_sft_trainer \
    data.train_files=./data/coder/train.parquet \
    data.val_files=./data/coder/val.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.train_batch_size=$bz \
    data.max_length=$max_length \
    optim.lr=$lr \
    data.prompt_dict_keys=['qwen_prompt'] \
    data.response_dict_keys=['answer'] \
    data.use_original_prompt=True \
    data.micro_batch_size_per_gpu=$micro_batch_size \
    model.partial_pretrain=Qwen/Qwen2.5-Coder-7B \
    model.use_liger=True \
    model.fsdp_config.model_dtype=bf16 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$project_name \
    trainer.experiment_name="$experiment_name-$(date +%Y%m%d-%H%M%S)" \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null \
    trainer.test_freq=100000 \
    trainer.save_freq=400 \
    trainer.total_epochs=1 \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true \
    trainer.objective_trans=logp
