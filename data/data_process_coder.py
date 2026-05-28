import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset


QWEN_PROMPT = (
    "<|im_start|>user\n{prompt}\n<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def build_row(item: dict) -> dict:
    prompt = item["problem"]
    response = item["solution"]
    return {
        "data_source": "Magicoder-OSS-Instruct-75K",
        "ability": "coder",
        "reward_model": {
            "style": "rule",
            "ground_truth": response,
        },
        "extra_info": {
            "prompt": prompt,
            "answer": response,
            "qwen_prompt": QWEN_PROMPT.format(prompt=prompt),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="ise-uiuc/Magicoder-OSS-Instruct-75K")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="data/coder")
    parser.add_argument("--train_file_name", type=str, default="train.parquet")
    parser.add_argument("--val_file_name", type=str, default="val.parquet")
    parser.add_argument("--validation_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_name, split=args.split)
    df = pd.DataFrame(build_row(item) for item in dataset)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_dir / args.train_file_name, index=False)

    val_size = min(args.validation_size, len(df))
    val_df = df.sample(n=val_size, random_state=args.seed)
    val_df.to_parquet(output_dir / args.val_file_name, index=False)

    print(f"Saved {len(df)} training examples to {output_dir / args.train_file_name}")
    print(f"Saved {val_size} validation examples to {output_dir / args.val_file_name}")


if __name__ == "__main__":
    main()
