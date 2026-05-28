# Coder Evaluation

This wrapper evaluates Qwen2.5-Coder-7B checkpoints with EvalPlus on HumanEval
and MBPP, matching the coder benchmark commands used in development.

Install EvalPlus in your environment, then run:

```bash
pip install -r evaluations/coder/requirements.txt
```

```bash
python evaluations/coder/main.py \
    --model ./checkpoints/coder/qwen2.5-coder-7b-lr-5e-5-bz-256-max_length-3096-nproc_per_node-2-micro_batch_size-4-original/global_step_293 \
    --tp 2
```

By default, both `humaneval` and `mbpp` are evaluated with vLLM greedy decoding.
