# Instruction Tuning Evaluation

This directory contains a thin wrapper around the AlpacaEval generation and judging flow used for the Mix-Magpie-EvoInstruct-140K instruction-tuning runs.

Generate AlpacaEval model outputs:

```bash
python evaluations/instruction_tuning/main.py \
    --model_path ./checkpoints/instruction_tuning/qwen2.5-7b-lr-5e-6-bz-256-max_length-2048-nproc_per_node-2-weight_decay-1e-4-micro_batch_size-8-original/global_step_545 \
    --model_name qwen2.5-7b-mix-magpie-evoinstruct-140k-original \
    --tensor_parallel_size 2
```

Run AlpacaEval judging after generation:

```bash
export OPENAI_API_KEY=...
python evaluations/instruction_tuning/main.py \
    --model_path ./checkpoints/instruction_tuning/qwen2.5-7b-lr-5e-6-bz-256-max_length-2048-nproc_per_node-2-weight_decay-1e-4-micro_batch_size-8-original/global_step_545 \
    --model_name qwen2.5-7b-mix-magpie-evoinstruct-140k-original \
    --tensor_parallel_size 2 \
    --run_alpaca_eval
```

For pairwise comparisons, pass `--reference_outputs path/to/reference/output.json`.
