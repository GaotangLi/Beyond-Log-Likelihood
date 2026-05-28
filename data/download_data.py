import os
from pathlib import Path

from datasets import load_dataset


LOW_RESOURCE_LANGUAGE_REPO = os.environ.get("BLL_LOW_RESOURCE_LANGUAGE_REPO", "gaotang/low_resource_language")
LOW_RESOURCE_LANGUAGE_DATA_FILES = {
    "train": "data/train.parquet",
    "validation": "data/validation.parquet",
    "test": "data/test_data.json",
}
CODER_REPO = os.environ.get("BLL_CODER_REPO", "gaotang/coder_sft")
CODER_DATA_FILES = {
    "train": "data/train.parquet",
    "validation": "data/validation.parquet",
}
INSTRUCTION_TUNING_REPO = os.environ.get("BLL_INSTRUCTION_TUNING_REPO", "gaotang/mix_magpie_evol_instruct_140k")
INSTRUCTION_TUNING_DATA_FILES = {
    "train": "data/train.parquet",
    "validation": "data/validation.parquet",
}
KNOWLEDGE_MEMORIZATION_REPO = os.environ.get("BLL_KNOWLEDGE_MEMORIZATION_REPO", "gaotang/knowledge_memorization_openbookqa")
KNOWLEDGE_MEMORIZATION_DATA_FILES = {
    "train_clean": "data/train_clean.parquet",
    "validation_clean": "data/validation_clean.parquet",
    "train_noisy": "data/train_noisy_0.3.parquet",
    "validation_noisy": "data/validation_noisy_0.3.parquet",
    "test": "data/test_data.json",
}

ds = load_dataset("gaotang/numina-cot-subset-67k", split="train")
math_path = Path("./data/math") 
df = ds.to_parquet(math_path / "train.parquet")

ds = load_dataset("gaotang/numina-cot-subset-67k", split="train")
ds_subset = ds.select(range(128))
ds_subset.to_parquet(math_path / "val.parquet")

ds = load_dataset("gaotang/medical_sft_processed", split="train")
medical_path = Path("./data/medical")
medical_path.mkdir(parents=True, exist_ok=True)
df = ds.to_parquet(medical_path / "train.parquet")

ds = load_dataset("gaotang/medical_sft_processed", split="train")
ds_subset = ds.select(range(128))  # This keeps it as a Dataset object
ds_subset.to_parquet(medical_path / "val.parquet")

figfont_path = Path("./data/figfont")
figfont_path.mkdir(parents=True, exist_ok=True)
ds = load_dataset("gaotang/figlet_font", split="train")
df = ds.to_parquet(figfont_path / "train.parquet")

ds = load_dataset("gaotang/figlet_font", split="train")
ds_subset = ds.select(range(128))  # This keeps it as a Dataset object
ds_subset.to_parquet(figfont_path / "val.parquet")

test_path = Path("./evaluations/figfont/data")
test_path.mkdir(parents=True, exist_ok=True)
ds = load_dataset("gaotang/figlet_font", split="test")
df = ds.to_parquet(test_path / "test.parquet")

low_resource_language_path = Path("./data/low_resource_language")
low_resource_language_path.mkdir(parents=True, exist_ok=True)
low_resource_language_eval_path = Path("./evaluations/low_resource_language/data")
low_resource_language_eval_path.mkdir(parents=True, exist_ok=True)

train_val_files = {
    "train": LOW_RESOURCE_LANGUAGE_DATA_FILES["train"],
    "validation": LOW_RESOURCE_LANGUAGE_DATA_FILES["validation"],
}
ds = load_dataset(
    LOW_RESOURCE_LANGUAGE_REPO,
    data_files=train_val_files,
    split="train",
)
ds.to_parquet(low_resource_language_path / "train.parquet")

ds = load_dataset(
    LOW_RESOURCE_LANGUAGE_REPO,
    data_files=train_val_files,
    split="validation",
)
ds.to_parquet(low_resource_language_path / "val.parquet")

ds = load_dataset(
    LOW_RESOURCE_LANGUAGE_REPO,
    data_files={"test": LOW_RESOURCE_LANGUAGE_DATA_FILES["test"]},
    split="test",
)
ds.to_json(low_resource_language_eval_path / "test_data.json", orient="records", lines=False)

coder_path = Path("./data/coder")
coder_path.mkdir(parents=True, exist_ok=True)
ds = load_dataset(
    CODER_REPO,
    data_files=CODER_DATA_FILES,
    split="train",
)
ds.to_parquet(coder_path / "train.parquet")

ds = load_dataset(
    CODER_REPO,
    data_files=CODER_DATA_FILES,
    split="validation",
)
ds.to_parquet(coder_path / "val.parquet")

instruction_tuning_path = Path("./data/instruction_tuning")
instruction_tuning_path.mkdir(parents=True, exist_ok=True)
ds = load_dataset(
    INSTRUCTION_TUNING_REPO,
    data_files=INSTRUCTION_TUNING_DATA_FILES,
    split="train",
)
ds.to_parquet(instruction_tuning_path / "train.parquet")

ds = load_dataset(
    INSTRUCTION_TUNING_REPO,
    data_files=INSTRUCTION_TUNING_DATA_FILES,
    split="validation",
)
ds.to_parquet(instruction_tuning_path / "val.parquet")

knowledge_memorization_path = Path("./data/knowledge_memorization")
knowledge_memorization_path.mkdir(parents=True, exist_ok=True)
knowledge_memorization_eval_path = Path("./evaluations/knowledge_memorization/data")
knowledge_memorization_eval_path.mkdir(parents=True, exist_ok=True)

ds = load_dataset(
    KNOWLEDGE_MEMORIZATION_REPO,
    data_files={
        "train_clean": KNOWLEDGE_MEMORIZATION_DATA_FILES["train_clean"],
        "validation_clean": KNOWLEDGE_MEMORIZATION_DATA_FILES["validation_clean"],
        "train_noisy": KNOWLEDGE_MEMORIZATION_DATA_FILES["train_noisy"],
        "validation_noisy": KNOWLEDGE_MEMORIZATION_DATA_FILES["validation_noisy"],
    },
    split="train_clean",
)
ds.to_parquet(knowledge_memorization_path / "train_clean.parquet")

ds = load_dataset(
    KNOWLEDGE_MEMORIZATION_REPO,
    data_files={
        "train_clean": KNOWLEDGE_MEMORIZATION_DATA_FILES["train_clean"],
        "validation_clean": KNOWLEDGE_MEMORIZATION_DATA_FILES["validation_clean"],
        "train_noisy": KNOWLEDGE_MEMORIZATION_DATA_FILES["train_noisy"],
        "validation_noisy": KNOWLEDGE_MEMORIZATION_DATA_FILES["validation_noisy"],
    },
    split="validation_clean",
)
ds.to_parquet(knowledge_memorization_path / "val_clean.parquet")

ds = load_dataset(
    KNOWLEDGE_MEMORIZATION_REPO,
    data_files={
        "train_clean": KNOWLEDGE_MEMORIZATION_DATA_FILES["train_clean"],
        "validation_clean": KNOWLEDGE_MEMORIZATION_DATA_FILES["validation_clean"],
        "train_noisy": KNOWLEDGE_MEMORIZATION_DATA_FILES["train_noisy"],
        "validation_noisy": KNOWLEDGE_MEMORIZATION_DATA_FILES["validation_noisy"],
    },
    split="train_noisy",
)
ds.to_parquet(knowledge_memorization_path / "train_noisy_0.3.parquet")

ds = load_dataset(
    KNOWLEDGE_MEMORIZATION_REPO,
    data_files={
        "train_clean": KNOWLEDGE_MEMORIZATION_DATA_FILES["train_clean"],
        "validation_clean": KNOWLEDGE_MEMORIZATION_DATA_FILES["validation_clean"],
        "train_noisy": KNOWLEDGE_MEMORIZATION_DATA_FILES["train_noisy"],
        "validation_noisy": KNOWLEDGE_MEMORIZATION_DATA_FILES["validation_noisy"],
    },
    split="validation_noisy",
)
ds.to_parquet(knowledge_memorization_path / "val_noisy_0.3.parquet")

ds = load_dataset(
    KNOWLEDGE_MEMORIZATION_REPO,
    data_files={"test": KNOWLEDGE_MEMORIZATION_DATA_FILES["test"]},
    split="test",
)
ds.to_json(knowledge_memorization_eval_path / "test_data.json", orient="records", lines=False)
