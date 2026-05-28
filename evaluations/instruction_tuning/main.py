import argparse
import json
import subprocess
from pathlib import Path

import datasets
from tqdm import tqdm
from vllm import LLM, SamplingParams


QWEN_PROMPT_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
DEFAULT_STOP_TOKENS = ["<|im_end|>", "<endoftext>", "\n\n\n\n", "[END_OF_TEXT]"]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate AlpacaEval outputs with vLLM and optionally run AlpacaEval judging.")
    parser.add_argument("--model_path", required=True, help="Local checkpoint path or Hugging Face model ID.")
    parser.add_argument("--model_name", required=True, help="Generator name written into the AlpacaEval output file.")
    parser.add_argument("--output_file", default=None, help="Output JSON path. Defaults to results/instruction_tuning/<model_name>/output.json.")
    parser.add_argument("--dataset_name", default="tatsu-lab/alpaca_eval")
    parser.add_argument("--dataset_config", default="alpaca_eval")
    parser.add_argument("--dataset_split", default="eval")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--stop_tokens", nargs="+", default=DEFAULT_STOP_TOKENS)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--prompt_template", default=QWEN_PROMPT_TEMPLATE)
    parser.add_argument("--run_alpaca_eval", action="store_true", help="Run alpaca_eval after output generation.")
    parser.add_argument("--annotators_config", default="weighted_alpaca_eval_gpt4_turbo")
    parser.add_argument("--reference_outputs", default=None, help="Optional reference outputs for pairwise AlpacaEval comparisons.")
    return parser.parse_args()


def generate_outputs(args, output_file: Path) -> None:
    eval_set = datasets.load_dataset(args.dataset_name, args.dataset_config)[args.dataset_split]
    if args.num_samples is not None:
        eval_set = eval_set.select(range(min(args.num_samples, len(eval_set))))

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k if args.top_k > 0 else -1,
        max_tokens=args.max_tokens,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        stop=args.stop_tokens,
    )

    prompts = [args.prompt_template.format(prompt=example["instruction"]) for example in eval_set]
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for example, output in tqdm(zip(eval_set, outputs), total=len(eval_set), desc="Formatting outputs"):
        results.append(
            {
                "instruction": example["instruction"],
                "output": output.outputs[0].text.strip(),
                "generator": args.model_name,
                "dataset": example["dataset"],
                "datasplit": "eval",
            }
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(results)} AlpacaEval outputs to {output_file}")


def run_alpaca_eval(args, output_file: Path) -> None:
    command = [
        "alpaca_eval",
        "--model_outputs",
        str(output_file),
        "--annotators_config",
        args.annotators_config,
    ]
    if args.reference_outputs:
        command.extend(["--reference_outputs", args.reference_outputs])
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    output_file = Path(args.output_file) if args.output_file else Path("results") / "instruction_tuning" / args.model_name / "output.json"
    generate_outputs(args, output_file)
    if args.run_alpaca_eval:
        run_alpaca_eval(args, output_file)


if __name__ == "__main__":
    main()
