import argparse
import json
import random
from pathlib import Path

import pandas as pd
from datasets import load_dataset


LLAMA_PROMPT = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)
QWEN_PROMPT = (
    "<|im_start|>user\n{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
CHOICES = ("A", "B", "C", "D")


def build_prompt(example: dict) -> str:
    options = example["choices"]["text"]
    return (
        "Please complete the following multiple choice question.\n"
        f"Question: {example['question_stem']}\n"
        "Options:\n"
        f"A. {options[0]}\n"
        f"B. {options[1]}\n"
        f"C. {options[2]}\n"
        f"D. {options[3]}"
    )


def build_train_row(example: dict, answer: str) -> dict:
    prompt = build_prompt(example)
    return {
        "data_source": "openbookqa",
        "ability": "knowledge memorization",
        "reward_model": {
            "style": "rule",
            "ground_truth": answer,
        },
        "extra_info": {
            "prompt": prompt,
            "answer": answer,
            "llama_prompt": LLAMA_PROMPT.format(prompt=prompt),
            "qwen_prompt": QWEN_PROMPT.format(prompt=prompt),
        },
    }


def build_test_row(example: dict) -> dict:
    return {
        "prompt": build_prompt(example),
        "answer": example["answerKey"],
    }


def build_train_frame(train_ds, val_ds, noise_rate: float, rng: random.Random) -> pd.DataFrame:
    train_rows = []
    noisy_indices = set(rng.sample(range(len(train_ds)), int(len(train_ds) * noise_rate)))

    for idx, example in enumerate(train_ds):
        answer = example["answerKey"]
        if idx in noisy_indices:
            answer = rng.choice(list(set(CHOICES) - {answer}))
        train_rows.append(build_train_row(example, answer))

    for example in val_ds:
        train_rows.append(build_train_row(example, example["answerKey"]))

    return pd.DataFrame(train_rows)


def write_split(df: pd.DataFrame, output_dir: Path, suffix: str, validation_size: int, seed: int) -> None:
    train_path = output_dir / f"train_{suffix}.parquet"
    val_path = output_dir / f"val_{suffix}.parquet"
    df.to_parquet(train_path, index=False)
    df.sample(n=min(validation_size, len(df)), random_state=seed).to_parquet(val_path, index=False)
    print(f"Saved {len(df)} training examples to {train_path}")
    print(f"Saved validation examples to {val_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="allenai/openbookqa")
    parser.add_argument("--dataset_config", type=str, default="additional")
    parser.add_argument("--output_dir", type=str, default="data/knowledge_memorization")
    parser.add_argument("--eval_output_dir", type=str, default="evaluations/knowledge_memorization/data")
    parser.add_argument("--noise_rates", type=str, default="0.0,0.3")
    parser.add_argument("--validation_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_name, args.dataset_config)
    train_ds = dataset["train"]
    val_ds = dataset["validation"]
    test_ds = dataset["test"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_output_dir = Path(args.eval_output_dir)
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    test_data = [build_test_row(example) for example in test_ds]
    (eval_output_dir / "test_data.json").write_text(json.dumps(test_data, indent=4, ensure_ascii=False), encoding="utf-8")

    for raw_noise_rate in args.noise_rates.split(","):
        noise_rate = float(raw_noise_rate.strip())
        suffix = "clean" if noise_rate == 0.0 else f"noisy_{noise_rate}"
        df = build_train_frame(train_ds, val_ds, noise_rate, random.Random(args.seed))
        write_split(df, output_dir, suffix, args.validation_size, args.seed)


if __name__ == "__main__":
    main()
