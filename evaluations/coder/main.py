import argparse
import subprocess
import sys


def parse_datasets(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model id or local checkpoint path.")
    parser.add_argument("--datasets", default="humaneval,mbpp")
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--greedy", action="store_true", default=True)
    parser.add_argument("--force_base_prompt", action="store_true", default=False)
    args = parser.parse_args()

    for dataset in parse_datasets(args.datasets):
        cmd = [
            sys.executable,
            "-m",
            "evalplus.evaluate",
            "--model",
            args.model,
            "--dataset",
            dataset,
            "--backend",
            args.backend,
            "--tp",
            str(args.tp),
        ]
        if args.greedy:
            cmd.append("--greedy")
        if args.force_base_prompt:
            cmd.append("--force-base-prompt")
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
