# Beyond Log Likelihood: Probability-Based Objectives for Supervised Fine-Tuning across the Model Capability Continuum


<!-- [**🤗 Huggingfacel**](xx) | [**📖 Paper**](yy)  -->

[**📖 Paper**](https://arxiv.org/abs/2510.00526)

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

The installation requirements are minimal. The main dependencies are:

```bash
verl(==0.4.0.dev0)
torch
vllm
flash_attn
```

---

## 🚀 Training

The training scrips are provided in [`scripts/training`](`scripts/training`).  

<!-- Available objectives include. -->
<!-- The key argument to modify is `model.partial_pretrain` and `trainer.objective_trans`. -->

---

## 📊 Evaluation 

The evaluation scripts are provided in [`scripts/evaluation`](`scripts/evaluation`).

<!-- TODO: More explanations -->


---

## 📑 Datasets

The dataset processing and downloading code are provided in [`data`](`data/`).

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