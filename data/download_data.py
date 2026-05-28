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
