import argparse
import json
import random
import re
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


LLAMA_PROMPT = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)
QWEN_PROMPT = (
    "<|im_start|>user\n{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument(
        "--eval_file",
        type=str,
        default="evaluations/low_resource_language/data/test_data.json",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--model_save_name", type=str, required=True)
    parser.add_argument("--output_file_name", type=str, default="raw_results")
    parser.add_argument("--output_dir", type=str, default="results/low_resource_language")
    parser.add_argument("--choices", type=str, default="ABCDEFGHIJ")
    parser.add_argument(
        "--answer_strategy",
        type=str,
        default="most_explicit",
        choices=["most_explicit", "last", "first", "most_common"],
    )
    parser.add_argument(
        "--max_saved_predictions",
        type=int,
        default=500,
        help="Save a deterministic sample of raw predictions. Use -1 to save all.",
    )
    parser.add_argument("--prediction_sample_seed", type=int, default=0)
    return parser.parse_args()


def get_model(args):
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
    return model, tokenizer


def load_eval_data(eval_file: str, debug: bool):
    with open(eval_file, "r", encoding="utf-8") as f:
        input_data = json.load(f)
    if debug:
        return input_data[:10]
    return input_data


def build_prompts(model_name: str, input_data: Iterable[dict]):
    input_prompts = [item["prompt"] for item in input_data]
    model_name_lower = model_name.lower()
    if "llama" in model_name_lower:
        return [LLAMA_PROMPT.format(prompt=item) for item in input_prompts]
    if "qwen" in model_name_lower:
        return [QWEN_PROMPT.format(prompt=item) for item in input_prompts]
    raise NotImplementedError(f"Model {model_name} is not supported")


def extract_mc_answer(
    text: str,
    valid_choices: list[str],
    case_sensitive: bool = False,
    strategy: str = "most_explicit",
) -> str | None:
    if not text:
        return None

    original_choices = valid_choices
    if not case_sensitive:
        text = text.upper()
        valid_choices = [choice.upper() for choice in valid_choices]

    sorted_choices = sorted(valid_choices, key=len, reverse=True)
    choice_pattern = "|".join(re.escape(choice) for choice in sorted_choices)
    before = r"(?<![A-Za-z0-9])"
    after = r"(?![A-Za-z0-9])"
    standalone = rf"{before}({choice_pattern}){after}"

    patterns_with_scores = [
        (rf"[(\[{{\"']\s*({choice_pattern})\s*[)\]}}\"']", 10),
        (rf"{before}({choice_pattern})\s*[).,\]:;!?\u3002\uff09\uff0e\uff1a\uff1b\uff01\uff1f]", 9),
        (rf"[(\[{{]\s*({choice_pattern}){after}", 8),
        (rf"[:=\-]\s*({choice_pattern})\s*[:=\-]", 8),
        (rf"{before}({choice_pattern})\s*[:=\-]", 7),
        (rf"[:=\->]\s*({choice_pattern}){after}", 7),
        (rf"{before}({choice_pattern})\s*(?:[,.;!?]|$)", 6),
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

    if strategy == "first":
        result = min(matches, key=lambda item: item["position"])["choice"]
    elif strategy == "last":
        result = max(matches, key=lambda item: item["position"])["choice"]
    elif strategy == "most_common":
        counts = Counter(match["choice"] for match in matches)
        top_count = counts.most_common(1)[0][1]
        top_choices = {choice for choice, count in counts.items() if count == top_count}
        result = max(
            [match for match in matches if match["choice"] in top_choices],
            key=lambda item: (item["score"], item["position"]),
        )["choice"]
    elif strategy == "most_explicit":
        result = max(matches, key=lambda item: (item["score"], item["position"]))["choice"]
    else:
        raise ValueError(f"Unknown answer extraction strategy: {strategy}")

    if not case_sensitive:
        for original, normalized in zip(original_choices, valid_choices):
            if normalized == result:
                return original
    return result


def write_outputs(output_data, lang_scores, args, total_score, total_count):
    output_path = Path(args.output_dir) / args.model_save_name / f"{args.output_file_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.max_saved_predictions < 0 or len(output_data) <= args.max_saved_predictions:
        saved_predictions = output_data
    else:
        rng = random.Random(args.prediction_sample_seed)
        saved_predictions = rng.sample(output_data, args.max_saved_predictions)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(saved_predictions, f, indent=4, ensure_ascii=False)

    score_json = {}
    for lang, values in sorted(lang_scores.items()):
        score_json[lang] = {
            "acc": values["score"] / values["total"],
            "count": values["total"],
            "total_score": values["score"],
        }
    score_json["average"] = float(np.mean([values["acc"] for values in score_json.values()]))
    score_json["overall"] = {
        "acc": total_score / total_count,
        "count": total_count,
        "total_score": total_score,
    }

    with open(output_path.parent / f"{args.output_file_name}_score.json", "w", encoding="utf-8") as f:
        json.dump(score_json, f, indent=4, ensure_ascii=False)


def main():
    args = get_args()
    model, _ = get_model(args)
    input_data = load_eval_data(args.eval_file, args.debug)
    input_prompts = build_prompts(args.model_name, input_data)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    outputs = model.generate(input_prompts, sampling_params)
    outputs = [output.outputs[0].text for output in outputs]

    valid_choices = list(args.choices)
    lang_scores = defaultdict(lambda: defaultdict(int))
    output_data = []
    total_score = 0
    total_count = 0

    for item, output in zip(input_data, outputs):
        item_copy = deepcopy(item)
        extracted = extract_mc_answer(
            output,
            valid_choices=valid_choices,
            strategy=args.answer_strategy,
        )
        score = extracted is not None and extracted.lower() == item_copy["answer"].lower()

        item_copy["output"] = output
        item_copy["extracted_output_option"] = extracted
        item_copy["score"] = score
        output_data.append(item_copy)

        total_score += score
        total_count += 1
        lang_scores[item_copy["lang"]]["total"] += 1
        lang_scores[item_copy["lang"]]["score"] += score

    print(f"Total score: {total_score / total_count}")
    print(f"Total count: {total_count}, total length: {len(input_data)}, total score: {total_score}")

    write_outputs(output_data, lang_scores, args, total_score, total_count)

    if args.debug:
        for output in outputs:
            print(output)
            print("-" * 100)
            print()


if __name__ == "__main__":
    main()
