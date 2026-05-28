# Low-Resource Language Evaluation

This evaluator runs multilingual multiple-choice evaluation on the MMLU-ProX-style
test set generated for the model-weak multilingual setting.

```bash
python evaluations/low_resource_language/main.py \
    --model_name Qwen/Qwen2.5-7B \
    --model_save_name qwen-2.5-7b \
    --tensor_parallel_size 2 \
    --max_tokens 1024 \
    --output_file_name zero_shot
```

The full `test_data.json` file is intentionally ignored by git because it is
large. Use `python data/download_data.py` to download it with the rest of the
repository data.
