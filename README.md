# Beyond Log Likelihood: Probability-Based Objectives for Supervised Fine-Tuning across the Model Capability Continuum


[**🤗 Huggingfacel**](https://huggingface.co/collections/gaotang/beyond-log-likelihood-68ddd78e78cb1e2f6b885a4e) | [**📖 Paper**](https://arxiv.org/abs/2510.00526) 

---

## 📑 Table of Contents
- [📖 Overview](#-overview)  
- [📂 Repository Structure](#-repository-structure)  
- [⚙️ Installation](#️-installation)  
- [🚀 Training](#-training)  
- [📊 Evaluation](#-evaluation)  
- [📑 Datasets](#-datasets)  
- [🙏 Acknowledgements](#-acknowledgements)  
- [📚 Citation](#-citation)  

---

## 📖 Overview

Supervised fine-tuning (SFT) is the standard post-training approach for large language models (LLMs), but its default objective — **Negative Log-Likelihood (NLL)** — is not universally optimal. While NLL is classically optimal when training from scratch, post-training operates in a different paradigm and could violate its optimality assumptions, where models already encode task-relevant priors and supervision can be long and noisy.  

In addition, language models are trained to be *general-purpose* models, but the vast differences between downstream tasks mean that they should not be treated equally. Tasks differ in how much useful prior knowledge is already encoded from pretraining, and thus a single objective may not work well across all cases.  

To this end, we study a **general family of probability-based objectives** and characterize their effectiveness under different conditions. We first **categorize objectives** based on how they distribute gradient weight:  
- **Prior-leaning objectives:** Emphasize mid- to high-probability tokens (e.g., −p, −p¹⁰, thresholded variants), leveraging model priors to refine already plausible predictions.  
- **Prior-averse objectives:** Emphasize low-probability tokens (e.g., −log p), encouraging the model to learn broadly even when priors are weak or misaligned.  

Building on this categorization, we introduce the **model-capability continuum** that characterizes the effectiveness of different objectives:  
- **Model-Strong (MS):** Base models already encode strong priors (e.g., math). Prior-leaning objectives consistently outperform NLL by focusing on reliable signals.  
- **Model-Intermediate (MI):** Models have partial priors (e.g., medical reasoning). No single objective dominates; performance depends on data and supervision.  
- **Model-Weak (MW):** Models lack useful priors (e.g., novel puzzles). NLL remains superior by enforcing learning from low-probability tokens.  

This framework provides a principled view of when and why different SFT objectives succeed or fail.

---

## 📂 Repository Structure

```text
Beyond-Log-Likelihood/
│
├── data/                     # Data processing files
│   ├── data_process_figfont.py
│   ├── data_process_math.py
│   ├── data_process_medical.py
│   └── download_data.py
│
├── evaluations/              # Evaluation pipelines for different tasks
│   ├── figfont/
│   ├── math/
│   └── medical/
│
├── main_verl/                # Core training framework
│   ├── trainer/
│   │   ├── config/
│   │   └── fsdp_sft_trainer.py   # Main Trainer
│   └── utils/
│
├── scripts/                  # Scripts for running experiments
│   ├── evaluation/
│   └── training/
│   └── one_click/  # train and eva in one click with your passed-in parameters
│
├── .gitignore
└── README.md
```

---

## ⚙️ Installation 

The installation requirements are minimal. (You may use your own environments for running the code.) The main dependencies are:

```bash
verl(==0.4.0.dev0)
torch
vllm
flash_attn
```

Before training, run the following code to download all necessary data (or you can generate your own training data by following files inside [📑 Datasets](#-datasets)):

```bash
python data/download_data.py
```


---

<!-- ## 🚀 Training

Training scripts are provided in [`scripts/training/`](scripts/training/). Each dataset has exemplar `.sh` files for quick use. In addition, we provide a **one-shot script** that automatically generates and runs the training command.

### One-Shot Training & Evaluation

To run training and evaluation in one step, use:

```bash
python scripts/one_click/script_generator.py \
    --dataset $DATASET \
    --model_save_name $MODEL_KEY \
    --trainer_objective_trans $OBJECTIVE \
    (--run_script)
```

where `$Dataset` specifies the dataset of use, which can be chosen from `[math, medical, figfont]`. `$model_save_name` specifies the model to be used, 
which can be chosen from the key of the following dictionary mappings:

```python
MODEL_MAPPING = {
    "qwen-2.5-math-1.5b": "Qwen/Qwen2.5-Math-1.5B",
    "qwen-2.5-math-7b": "Qwen/Qwen2.5-Math-7B",
    "qwen-2.5-1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    "deepseek-math-7b": "deepseek-ai/deepseek-math-7b-base",
}
```



The argument `--run_script` is a boolean argument that will directly run the code if specified. You may also specify the number of GPUs to be used by specifying `$nproc_per_node`. You can specify the specific GPU device by entering `--cuda_visible_devices $YOUR_DEVICES`. The **most important** argument is the objective, currently supported by the following keys in the dicitonary with explanations as values:

```python
{
  "original": "Original implementation of SFT",
  "GeneralFamily-alpha": "The function $(1-p^{\alpha})/\alpha$ where \alpha needs to be specified. A greater positive \alpha means the objective is more prior-leaning; and vice versa for prior-averse",
  "p": "1-p",
  "OnlyTopP-q": "The thresholded function (1-p) * 1[p >= q] (q to be specified)",
  "OnlyBottomP-q": "The thresholded function (1-p) * 1[p <= q] (q to be specified)",
  "OnlyTopLogP-q": "The thresholded funtion -log(p) * 1[p >= q] (q to be specified)", 
  "OnlyBottomLogP-q": "The thresholded function -log(p) * 1[p <= q] (q to be specified)" 
}
```


Specific examples (for each dataset) include the following:

```bash
python scripts/one_click/script_generator.py --dataset medical --model_save_name qwen-2.5-math-1.5b --trainer_objective_trans GeneralFamily-8 --run_script
python scripts/one_click/script_generator.py --dataset medical --model_save_name qwen-2.5-1.5b --trainer_objective_trans original --run_script
python scripts/one_click/script_generator.py --dataset figfont --model_save_name qwen-2.5-7b --trainer_objective_trans original --run_script
``` -->

## 🚀 Training

Training scripts are provided in [`scripts/training/`](scripts/training/). Each dataset has exemplar `.sh` files for quick use. In addition, we provide a **one-shot script** that automatically generates and runs the training command.

### One-Shot Training & Evaluation

To run training and evaluation in one step, use:

```bash
python scripts/one_click/script_generator.py \
    --dataset $DATASET \
    --model_save_name $MODEL_KEY \
    --trainer_objective_trans $OBJECTIVE \
    (--run_script)
```

#### Arguments

- **`--dataset`**: Specifies the dataset to use. Choose from: `[math, medical, figfont]`

- **`--model_save_name`**: Specifies the model key from the mapping below:

```python
MODEL_MAPPING = {
    "qwen-2.5-math-1.5b": "Qwen/Qwen2.5-Math-1.5B",
    "qwen-2.5-math-7b": "Qwen/Qwen2.5-Math-7B",
    "qwen-2.5-1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    "deepseek-math-7b": "deepseek-ai/deepseek-math-7b-base",
}
```

- **`--trainer_objective_trans`**: The most important argument. Specifies the training objective from the following options:

| Key | Description |
|-----|-------------|
| `original` | Original implementation of SFT |
| `GeneralFamily-alpha` | The function $(1-p^{\alpha})/\alpha$ where $\alpha$ needs to be specified. A greater positive $\alpha$ means the objective is more prior-leaning; and vice versa for prior-averse |
| `p` | $1-p$ |
| `OnlyTopP-q` | The thresholded function $(1-p) \cdot \mathbb{1}[p \geq q]$ ($q$ to be specified) |
| `OnlyBottomP-q` | The thresholded function $(1-p) \cdot \mathbb{1}[p \leq q]$ ($q$ to be specified) |
| `OnlyTopLogP-q` | The thresholded function $-\log(p) \cdot \mathbb{1}[p \geq q]$ ($q$ to be specified) |
| `OnlyBottomLogP-q` | The thresholded function $-\log(p) \cdot \mathbb{1}[p \leq q]$ ($q$ to be specified) |

- **`--run_script`**: (Optional) Boolean flag. If specified, directly executes the generated training command.

- **`--nproc_per_node`**: (Optional) Specifies the number of GPUs to use.

- **`--cuda_visible_devices`**: (Optional) Specifies specific GPU devices (e.g., `--cuda_visible_devices 0,1,2,3`).

#### Usage Examples

```bash
# Medical dataset with Qwen2.5-Math-1.5B using GeneralFamily objective (alpha=8)
python scripts/one_click/script_generator.py \
    --dataset medical \
    --model_save_name qwen-2.5-math-1.5b \
    --trainer_objective_trans GeneralFamily-8 \
    --run_script

# Medical dataset with Qwen2.5-1.5B using original SFT
python scripts/one_click/script_generator.py \
    --dataset medical \
    --model_save_name qwen-2.5-1.5b \
    --trainer_objective_trans original \
    --run_script

# Figfont dataset with Qwen2.5-7B using original SFT
python scripts/one_click/script_generator.py \
    --dataset figfont \
    --model_save_name qwen-2.5-7b \
    --trainer_objective_trans original \
    --run_script
```


---

## 📊 Evaluation 

The evaluation scripts are provided in [`scripts/evaluation/`](scripts/evaluation/).


---

## 📑 Datasets

The dataset processing and downloading code are provided in [`data/`](data/).

---

## 🙏 Acknowledgements

The implementation of this repository is built upon [veRL](https://github.com/volcengine/verl) and [DFT](https://github.com/yongliang-wu/DFT). We sincerely appreciate the efforts of these teams for their contributions to open-source research and development.


---

## 📚 Citation

If you find this repository useful, please cite:


```bibtex
@misc{li2025beyond,
  title        = {Beyond Log Likelihood: Probability-Based Objectives for Supervised Fine-Tuning across the Model Capability Continuum},
  author       = {Li, Gaotang and Qiu, Ruizhong and Chen, Xiusi and Ji, Heng and Tong, Hanghang},
  year         = {2025},
  eprint       = {2510.00526},
  archivePrefix = {arXiv},
  doi          = {10.48550/arXiv.2510.00526}
}
```