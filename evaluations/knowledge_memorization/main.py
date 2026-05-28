import argparse
import json
import random
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


LLAMA_PROMPT = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
QWEN_PROMPT = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def extract_mc_answer(text: str, valid_choices: Optional[list[str]] = None, case_sensitive: bool = False) -> Optional[str]:
    if not text:
        return None
    if valid_choices is None:
        valid_choices = ["A", "B", "C", "D"]

    original_choices = valid_choices
    if not case_sensitive:
        text = text.upper()
        valid_choices = [choice.upper() for choice in valid_choices]

    choice_pattern = "|".join(re.escape(choice) for choice in sorted(valid_choices, key=len, reverse=True))
    standalone = rf"(?<![A-Za-z0-9À-ÿ])({choice_pattern})(?![A-Za-z0-9À-ÿ])"
    patterns_with_scores = [
        (rf"[(\[{{【「『«\"']\s*({choice_pattern})\s*[)\]}}】」』»\"']", 10),
        (rf"(?<![A-Za-z0-9À-ÿ])({choice_pattern})\s*[).。）】」\]}}]", 9),
        (rf"[(\[{{【「『]\s*({choice_pattern})(?![A-Za-z0-9À-ÿ])", 8),
        (rf"[:：\-]\s*({choice_pattern})\s*[:：\-]", 8),
        (rf"(?<![A-Za-z0-9À-ÿ])({choice_pattern})\s*[:：]", 7),
        (rf"[:：→➜>]\s*({choice_pattern})(?![A-Za-z0-9À-ÿ])", 7),
        (rf"(?<![A-Za-z0-9À-ÿ])({choice_pattern})\s*(?:[,，.。!！?？;；]|$)", 6),
        (rf"(?<![A-Za-z0-9À-ÿ])\s+({choice_pattern})\s+(?![A-Za-z0-9À-ÿ])", 5),
        (standalone, 4),
    ]

    matches = []
    seen_positions = set()
    for pattern, score in patterns_with_scores:
        for match in re.finditer(pattern, text):
            choice = match.group(1)
            position = match.start(1)
            if choice in valid_choices and position not in seen_positions:
                matches.append({"choice": choice, "score": score, "position": position})
                seen_positions.add(position)
    if not matches:
        return None

    counts = Counter(match["choice"] for match in matches)
    result = max(matches, key=lambda match: (match["score"], counts[match["choice"]], match["position"]))["choice"]
    if case_sensitive:
        return result
    for original, normalized in zip(original_choices, valid_choices):
        if normalized == result:
            return original
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-14B")
    parser.add_argument("--eval_file", type=str, default="evaluations/knowledge_memorization/data/test_data.json")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--model_save_name", type=str, required=True)
    parser.add_argument("--output_file_name", type=str, default="raw_results")
    parser.add_argument("--results_dir", type=str, default="results/knowledge_memorization")
    return parser.parse_args()


def format_prompts(model_name: str, prompts: list[str]) -> list[str]:
    if "llama" in model_name.lower():
        return [LLAMA_PROMPT.format(prompt=prompt) for prompt in prompts]
    if "qwen" in model_name.lower():
        return [QWEN_PROMPT.format(prompt=prompt) for prompt in prompts]
    raise NotImplementedError(f"Model {model_name} not supported")


def main():
    args = parse_args()
    model = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_data = json.loads(Path(args.eval_file).read_text(encoding="utf-8"))
    if args.debug:
        input_data = input_data[:10]
    input_prompts = format_prompts(args.model_name, [item["prompt"] for item in input_data])

    outputs = model.generate(input_prompts, SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens))
    outputs = [output.outputs[0].text for output in outputs]

    output_data = []
    correct = 0
    for item, output in zip(input_data, outputs):
        item_copy = deepcopy(item)
        item_copy["output"] = output
        item_copy["extracted_output_option"] = extract_mc_answer(output)
        item_copy["score"] = item_copy["extracted_output_option"] is not None and item_copy["extracted_output_option"].lower() == item_copy["answer"].lower()
        correct += int(item_copy["score"])
        output_data.append(item_copy)

    average = correct / len(output_data)
    print(f"Total score: {average}")
    print(f"Total count: {len(output_data)}, total length: {len(input_data)}, total score: {correct}")

    output_path = Path(args.results_dir) / args.model_save_name / f"{args.output_file_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(random.sample(output_data, min(500, len(output_data))), indent=4), encoding="utf-8")
    score_path = output_path.with_name(f"{args.output_file_name}_score.json")
    score_path.write_text(json.dumps({"average": average}, indent=4), encoding="utf-8")


if __name__ == "__main__":
    main()
