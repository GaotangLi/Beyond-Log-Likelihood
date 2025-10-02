import random
import os
import argparse
import time
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluate import evaluate
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor
from model_utils import load_hf_lm_and_tokenizer, generate_completions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="math_oai", type=str)
    parser.add_argument("--data_dir", default="evaluations/math/data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=4096, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=-1,
                        help="Number of GPUs to use for tensor parallelism. -1 means use all available GPUs.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90,
                        help="GPU memory utilization for vLLM.")
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    
    # Auto-detect number of GPUs if not specified
    if args.tensor_parallel_size == -1:
        args.tensor_parallel_size = torch.cuda.device_count()
        print(f"Auto-detected {args.tensor_parallel_size} GPUs for tensor parallelism")
    
    return args


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        # examples = random.sample(examples, min(args.num_test_sample, len(examples)))
        examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    return examples


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def main(args):
    """Main function that processes all data using multi-GPU setup"""
    set_seed(args.seed)
    
    # Load model once with tensor parallelism across all GPUs
    print(f"Loading model with tensor_parallel_size={args.tensor_parallel_size}")
    
    if args.use_vllm:
        llm = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=True,
            # max_model_len=8192,
        )
        tokenizer = None
        if args.apply_chat_template:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, trust_remote_code=True
            )
    else:
        # For non-vLLM case, we keep the original single-GPU setup
        # You might want to implement data parallelism here if needed
        llm, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            load_in_half=True,
            use_fast_tokenizer=True,
            use_safetensors=args.use_safetensors,
        )
    
    # Process each dataset
    data_list = args.data_names.split(",")
    results = []
    
    for data_name in data_list:
        examples = prepare_data(data_name, args)
        print("=" * 50)
        print("data:", data_name, " ,remain samples:", len(examples))
        if len(examples) > 0:
            print(f"example: {examples[0]}")
        
        # Process all examples at once
        result_json = process_dataset(llm, tokenizer, data_name, args, examples)
        results.append(result_json)
    
    # Add average results
    data_list.append("avg")
    results.append(
        {
            "acc": sum([result["acc"] for result in results]) / len(results),
        }
    )
    
    # Save summary
    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, "w") as f:
        summary = {data: result for data, result in zip(data_list, results)}
        json.dump(summary, f, indent=4)
    print(f"Summary saved to {summary_file}")


def process_dataset(llm, tokenizer, data_name, args, examples):
    """Process a single dataset using the loaded model"""
    
    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for example in tqdm(examples, total=len(examples), desc=f"Preparing {data_name}"):
        idx = example["idx"]

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # repeat n times
    input_prompts = [
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    ]

    if args.apply_chat_template:
        input_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in input_prompts
        ]

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>", "<｜end▁of▁sentence｜>"]

    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")
        
    if "qwen2" in args.model_name_or_path.lower():
        stop_token_ids=[151645, 151643]
    elif "deepseek" in args.model_name_or_path.lower():
        stop_token_ids=[100001]
    else:
        stop_token_ids=None

    # start inference
    print(f"Starting inference for {data_name} with {len(input_prompts)} prompts...")
    start_time = time.time()

    # get all outputs in one inference
    if args.use_vllm:
        outputs = llm.generate(
            input_prompts,
            SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens_per_call,
                n=1,
                stop=stop_words,
                stop_token_ids=stop_token_ids,
            ),
        )

        outputs = sorted(
            outputs, key=lambda x: int(x.request_id)
        )  # sort outputs by request_id
        outputs = [output.outputs[0].text for output in outputs]
    else:
        outputs = generate_completions(
            model=llm,
            tokenizer=tokenizer,
            prompts=input_prompts,
            max_new_tokens=args.max_tokens_per_call,
            batch_size=16,
            stop_id_sequences=stop_words,
        )

    assert len(outputs) == len(input_prompts)

    # remove input_prompt from output and clean stop words
    codes = []
    for i in range(len(input_prompts)):
        output = outputs[i].rstrip()
        code = output
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)

    # extract preds
    print(f"Executing code for {data_name}...")
    results = [
        run_execute(executor, code, args.prompt_type, data_name) for code in tqdm(codes, desc="Executing")
    ]
    time_use = time.time() - start_time

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
        result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        for j in range(len(preds)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                "A",
                "B",
                "C",
                "D",
                "E",
            ]:
                preds[j] = choice_answer_clean(code[j])
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                # remove any non-choice char
                preds[j] = "".join(
                    [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                )

        sample.pop("prompt")
        sample.update({"code": code, "pred": preds, "report": reports})
        all_samples.append(sample)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    result_file = os.path.join(args.output_dir, f"{data_name}_samples.json")
    with open(result_file, "w", encoding="utf8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    
    # Evaluate
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
    )
    
    # Save metrics
    metrics_file = os.path.join(args.output_dir, f"{data_name}_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(result_json, f, indent=4)
    
    print(f"{data_name} completed!")
    print(f"Time used: {int(time_use // 60)}:{int(time_use % 60):02d}")
    print(f"Accuracy: {result_json['acc']:.4f}")
    print(f"Results saved to {result_file}")
    
    return result_json


if __name__ == "__main__":
    args = parse_args()
    main(args)