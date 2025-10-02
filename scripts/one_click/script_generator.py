from TEMPLATE import MATH_SCRIPT, MODEL_MAPPING, HYPERPARAMETER_MAPPING, MEDICAL_SCRIPT, FIGFONT_SCRIPT
import argparse 
from pathlib import Path 
import subprocess

selection = {
    "math": MATH_SCRIPT,
    "medical": MEDICAL_SCRIPT,
    "figfont": FIGFONT_SCRIPT,
}


def generate_script(args):
    dataset = args.dataset 
    model_save_name = args.model_save_name 
    base_model_official_path = MODEL_MAPPING[model_save_name]
    cuda_visible_devices = args.cuda_visible_devices 
    trainer_objective_trans = args.trainer_objective_trans 
    nproc_per_node = args.nproc_per_node 
    script = selection[dataset] 

    hyperparameter_mapping = HYPERPARAMETER_MAPPING[dataset][model_save_name] 


    if dataset == "math":
        script = script.format(
            nproc_per_node=nproc_per_node,
            model_save_name=model_save_name,
            base_model_official_path=base_model_official_path,
            cuda_visible_devices=cuda_visible_devices,
            trainer_objective_trans=trainer_objective_trans,
            lr=hyperparameter_mapping["lr"],
            case=hyperparameter_mapping["case"],
        )
    elif dataset == "medical": 
        script = script.format(
            nproc_per_node=nproc_per_node,
            model_save_name=model_save_name,
            base_model_official_path=base_model_official_path,
            cuda_visible_devices=cuda_visible_devices,
            trainer_objective_trans=trainer_objective_trans,
            lr=hyperparameter_mapping["lr"],
            prompt_dict_key=hyperparameter_mapping["prompt_dict_key"],
            response_dict_key=hyperparameter_mapping["response_dict_key"],
        )
    elif dataset == "figfont":  
        script = script.format(
            nproc_per_node=nproc_per_node,
            model_save_name=model_save_name,
            base_model_official_path=base_model_official_path,
            cuda_visible_devices=cuda_visible_devices,
            trainer_objective_trans=trainer_objective_trans,
            lr=hyperparameter_mapping["lr"],
            prompt_dict_key=hyperparameter_mapping["prompt_dict_key"],
            response_dict_key=hyperparameter_mapping["response_dict_key"],
        )
    else: 
        raise ValueError(f"Dataset {dataset} not supported.")
    
    return script 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["math", "medical", "figfont"])
    parser.add_argument("--nproc_per_node", type=int, default=2, help="Number of GPUs to use for training.") 
    parser.add_argument(
        "--model_save_name", 
        type=str, 
        required=True, 
        choices=["qwen-2.5-math-1.5b", "qwen-2.5-math-7b", "qwen-2.5-1.5b", "qwen-2.5-7b", "llama-3.1-8b", "llama-3.2-3b", "deepseek-math-7b"],
        help="The model to be used for training."
        "If you want to use a model that is not in the list, you can specify the base model path in the script."
        "Remember to also update the MODEL_MAPPING in TEMPLATE.py."     
    )

    parser.add_argument(
        "--trainer_objective_trans", 
        type=str, 
        required=True,
        default="original",
        # choices=["original", "p", "GeneralFamily-alpha", "OnlyTopP-q", "OnlyBottomP-q", "OnlyTopLogP-q", "OnlyBottomLogP-q"], q/alpha needs to be specified
        help="The objective transformation for the trainer. "
        "setting to 'original' will use the original SFT implementation." 
        "Available options: "
        "Original Implementation: original"
        "1-p: p"
        "(1-p^\alpha)/\alpha: GeneralFamily-alpha (alpha to be specified)" 
        "-p * 1[p >= q]: OnlyTopP-q (q to be specified)" 
        "-p * 1[p <= q]: OnlyBottomP-q (q to be specified)"  
        "-log(p) * 1[p >= q]: OnlyTopLogP-q (q to be specified)" 
        "-log(p) * 1[p <= q]: OnlyBottomLogP-q (q to be specified)"
    )

    parser.add_argument("--cuda_visible_devices", type=str, default="0,1") 
    parser.add_argument("--run_script", action="store_true", default=False)
    args = parser.parse_args()

    script = generate_script(args)

    SAVE_PATH = Path(f"scripts/one_click/{args.dataset}/{args.model_save_name}/{args.trainer_objective_trans}.sh")
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAVE_PATH.write_text(script)
    print(f"Script saved to {SAVE_PATH}")

    if args.run_script:
        subprocess.run(f"bash {SAVE_PATH}", shell=True)
    