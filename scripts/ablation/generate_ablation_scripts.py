#!/usr/bin/env python3
"""Generate ablation train-and-eval scripts."""

from __future__ import annotations

import os
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = ROOT / "scripts" / "ablation"

CONVEXITY_VALUES = [
    "0.1",
    "0.2",
    "0.3",
    "0.4",
    "0.5",
    "0.6",
    "0.7",
    "0.8",
    "0.9",
    "1.0",
    "2.0",
    "3.0",
    "4.0",
    "5.0",
    "6.0",
    "7.0",
    "8.0",
    "9.0",
    "10.0",
]

MODEL_SCALE_CONFIGS = [
    ("qwen-2.5-1.5b", "Qwen/Qwen2.5-1.5B", "4e-5", 2, 8),
    ("qwen-2.5-3b", "Qwen/Qwen2.5-3B", "4e-5", 2, 8),
    ("qwen-2.5-7b", "Qwen/Qwen2.5-7B", "4e-5", 2, 4),
    ("qwen-2.5-14b", "Qwen/Qwen2.5-14B", "2e-5", 2, 2),
    ("qwen-2.5-32b", "Qwen/Qwen2.5-32B", "1e-5", 4, 1),
    ("qwen-2.5-72b", "Qwen/Qwen2.5-72B", "1e-5", 4, 1),
]

PERCENTILE_THRESHOLDS = {
    0: "0",
    5: "0.038074225187301636",
    10: "0.17539461851119997",
    20: "0.5858418226242066",
    30: "0.8690263032913208",
    40: "0.970059871673584",
    50: "0.9936020970344543",
    60: "0.9985600709915161",
    70: "0.9996439814567566",
    80: "0.9999082088470459",
    90: "0.9999804496765137",
    100: "1.0",
}

TOP_PERCENTILES = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
BOTTOM_PERCENTILES = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
UNLIKEHOOD_TOP_PERCENTILES = [10, 20, 30, 40, 50, 60, 70, 80, 90]
UNLIKEHOOD_BOTTOM_PERCENTILES = [5, 20, 50, 80]


def default_cuda(nproc: int) -> str:
    return ",".join(str(i) for i in range(nproc))


def write_script(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.rstrip() + "\n", encoding="utf-8")
    os.chmod(path, 0o755)


def math_eval_block(result_group: str, model_save_name: str, checkpoint_step: int = 264) -> str:
    return dedent(
        f"""\

        PROMPT_TYPE="qwen-boxed"
        MODEL_SAVE_NAME="{model_save_name}"
        MODEL_NAME_OR_PATH="$save_path/global_step_{checkpoint_step}"
        OUTPUT_DIR="./results/ablation/{result_group}/$MODEL_SAVE_NAME/$experiment_name"
        n_sampling=16
        temperature=1

        SPLIT="test"
        NUM_TEST_SAMPLE=-1

        export CUDA_VISIBLE_DEVICES

        DATA_NAME="minerva_math,olympiadbench,aime24,amc23,math_oai"

        TOKENIZERS_PARALLELISM=false \\
        python3 -u evaluations/math/math_eval_multigpu.py \\
            --model_name_or_path ${{MODEL_NAME_OR_PATH}} \\
            --data_name ${{DATA_NAME}} \\
            --output_dir ${{OUTPUT_DIR}} \\
            --split ${{SPLIT}} \\
            --prompt_type ${{PROMPT_TYPE}} \\
            --num_test_sample ${{NUM_TEST_SAMPLE}} \\
            --seed 0 \\
            --temperature ${{temperature}} \\
            --n_sampling ${{n_sampling}} \\
            --top_p 1 \\
            --start 0 \\
            --end -1 \\
            --use_vllm \\
            --tensor_parallel_size ${{nproc_per_node}} \\
            --gpu_memory_utilization 0.95
        """
    )


def math_train_eval_script(
    *,
    result_group: str,
    model_save_name: str,
    model_path: str,
    lr: str,
    nproc: int,
    micro_batch_size: int,
    experiment_suffix: str,
    objective_trans: str,
    checkpoint_step: int = 264,
) -> str:
    return dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail

        CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-{default_cuda(nproc)}}}

        nproc_per_node={nproc}
        project_name=numina-cot
        lr={lr}
        bz=256
        max_length=3072
        micro_batch_size={micro_batch_size}

        experiment_name=numina-cot-{model_save_name}-lr-$lr-bz-$bz-max_length-$max_length-nproc_per_node-$nproc_per_node-micro_batch_size-$micro_batch_size-{experiment_suffix}
        save_path=./checkpoints/ablation/{result_group}/$experiment_name

        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \\
                -m main_verl.trainer.fsdp_sft_trainer \\
            data.train_files=./data/math/train.parquet \\
            data.val_files=./data/math/val.parquet \\
            data.prompt_key=extra_info \\
            data.response_key=extra_info \\
            data.train_batch_size=$bz \\
            data.max_length=$max_length \\
            optim.lr=$lr \\
            data.prompt_dict_keys=['question'] \\
            data.response_dict_keys=['answer'] \\
            data.micro_batch_size_per_gpu=$micro_batch_size \\
            model.partial_pretrain={model_path} \\
            model.use_liger=True \\
            model.fsdp_config.model_dtype=bf16 \\
            trainer.default_local_dir=$save_path \\
            trainer.project_name=$project_name \\
            trainer.experiment_name="$experiment_name-$(date +%Y%m%d-%H%M%S)" \\
            trainer.logger=['console','wandb'] \\
            trainer.default_hdfs_dir=null \\
            trainer.test_freq=100000 \\
            trainer.save_freq={checkpoint_step} \\
            trainer.total_epochs=1 \\
            ulysses_sequence_parallel_size=1 \\
            use_remove_padding=true \\
            trainer.objective_trans={objective_trans}
        """
    ) + math_eval_block(result_group, model_save_name, checkpoint_step)


def figfont_convexity_script(alpha: str) -> str:
    nproc = 2
    return dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail

        CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-{default_cuda(nproc)}}}

        nproc_per_node={nproc}
        project_name=figlet-font
        lr=5e-5
        bz=256
        max_length=800
        micro_batch_size=16
        weight_decay=1e-5

        experiment_name=qwen-2.5-7b-lr-$lr-bz-$bz-max_length-$max_length-nproc_per_node-$nproc_per_node-micro_batch_size-$micro_batch_size-weight_decay-$weight_decay-0916-convexity-{alpha}
        save_path=./checkpoints/ablation/convexity_figfont/$experiment_name

        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \\
                -m main_verl.trainer.fsdp_sft_trainer \\
            data.train_files=./data/figfont/train.parquet \\
            data.val_files=./data/figfont/val.parquet \\
            data.prompt_key=extra_info \\
            data.response_key=extra_info \\
            data.train_batch_size=$bz \\
            data.max_length=$max_length \\
            optim.lr=$lr \\
            data.prompt_dict_keys=['qwen_prompt'] \\
            data.response_dict_keys=['answer'] \\
            data.use_original_prompt=True \\
            data.micro_batch_size_per_gpu=$micro_batch_size \\
            model.partial_pretrain=Qwen/Qwen2.5-7B \\
            model.use_liger=True \\
            model.fsdp_config.model_dtype=bf16 \\
            optim.weight_decay=$weight_decay \\
            trainer.default_local_dir=$save_path \\
            trainer.project_name=$project_name \\
            trainer.experiment_name="$experiment_name-$(date +%Y%m%d-%H%M%S)" \\
            trainer.logger=['console','wandb'] \\
            trainer.default_hdfs_dir=null \\
            trainer.test_freq=100000 \\
            trainer.save_freq=264 \\
            trainer.total_epochs=1 \\
            ulysses_sequence_parallel_size=1 \\
            use_remove_padding=true \\
            trainer.objective_trans=convexity-{alpha}

        MODEL_SAVE_NAME="qwen-2.5-7b"
        MODEL_NAME_OR_PATH="$save_path/global_step_156"
        export CUDA_VISIBLE_DEVICES

        python evaluations/figfont/main.py \\
            --model_name ${{MODEL_NAME_OR_PATH}} \\
            --output_file_name ${{experiment_name}} \\
            --model_save_name ${{MODEL_SAVE_NAME}} \\
            --tensor_parallel_size ${{nproc_per_node}}
        """
    )


def generate_convexity_scripts() -> None:
    for alpha in CONVEXITY_VALUES:
        write_script(
            SCRIPT_ROOT / "convexity" / "figfont" / f"qwen-2.5-7b_convexity-{alpha}.sh",
            figfont_convexity_script(alpha),
        )
        write_script(
            SCRIPT_ROOT
            / "convexity"
            / "math"
            / f"qwen-2.5-math-1.5b_convexity-{alpha}.sh",
            math_train_eval_script(
                result_group="convexity_math",
                model_save_name="qwen-2.5-math-1.5b",
                model_path="Qwen/Qwen2.5-Math-1.5B",
                lr="5e-5",
                nproc=2,
                micro_batch_size=4,
                experiment_suffix=f"0916-convexity-{alpha}",
                objective_trans=f"convexity-{alpha}",
            ),
        )


def generate_model_scale_scripts() -> None:
    for model_save_name, model_path, lr, nproc, micro_batch_size in MODEL_SCALE_CONFIGS:
        for label, objective_trans in [("original", "original"), ("p", "p")]:
            write_script(
                SCRIPT_ROOT / "model_scale" / f"{model_save_name}_{label}.sh",
                math_train_eval_script(
                    result_group="model_scale",
                    model_save_name=model_save_name,
                    model_path=model_path,
                    lr=lr,
                    nproc=nproc,
                    micro_batch_size=micro_batch_size,
                    experiment_suffix=f"1113-{label}",
                    objective_trans=objective_trans,
                ),
            )


def generate_figure5_scripts() -> None:
    families = [
        ("top_p", "OnlyTopP", "OnlyTopP", TOP_PERCENTILES),
        ("bottom_p", "OnlyBottomP", "OnlyBottomP", BOTTOM_PERCENTILES),
        ("top_logp", "OnlyTopLogP", "OnlyTopLogP", TOP_PERCENTILES),
        ("bottom_logp", "OnlyBottomLogP", "OnlyBottomLogP", BOTTOM_PERCENTILES),
        ("unlikehood_top", "UnlikehoodTop", "UnlikehoodTop", UNLIKEHOOD_TOP_PERCENTILES),
        (
            "unlikehood_bottom",
            "UnlikehoodBottom",
            "UnlikehoodBottom",
            UNLIKEHOOD_BOTTOM_PERCENTILES,
        ),
    ]

    for group, suffix_prefix, objective_prefix, percentiles in families:
        for k in percentiles:
            threshold = PERCENTILE_THRESHOLDS[k]
            write_script(
                SCRIPT_ROOT
                / "figure5"
                / group
                / f"qwen-2.5-math-1.5b_{suffix_prefix}-K{k}.sh",
                math_train_eval_script(
                    result_group=f"figure5/{group}",
                    model_save_name="qwen-2.5-math-1.5b",
                    model_path="Qwen/Qwen2.5-Math-1.5B",
                    lr="5e-5",
                    nproc=2,
                    micro_batch_size=8,
                    experiment_suffix=f"figure5-{suffix_prefix}-K-{k}",
                    objective_trans=f"{objective_prefix}-{threshold}",
                ),
            )


def main() -> None:
    generate_convexity_scripts()
    generate_model_scale_scripts()
    generate_figure5_scripts()


if __name__ == "__main__":
    main()
