from datasets import load_dataset
from pathlib import Path 

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