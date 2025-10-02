#!/usr/bin/env python3
import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- Your dataset class (imported as requested)
try:
    from sft_dataset import SFTDataset_PRM
except Exception as e:
    raise RuntimeError(
        "Could not import SFTDataset_PRM from sft_dataset.py. "
        "Make sure it is on PYTHONPATH."
    ) from e


# -------------------------------
# Utilities
# -------------------------------

def tokenizers_compatible(tok_a, tok_b, sample: int = 2000) -> bool:
    """
    Heuristic check that both tokenizers share the exact same vocabulary & mapping.
    This is necessary for token-level KL.
    """
    if tok_a.vocab_size != tok_b.vocab_size:
        return False

    # Fast path: identical tokenizer name/path (often the same vocab)
    if getattr(tok_a, "name_or_path", None) == getattr(tok_b, "name_or_path", None):
        return True

    # Sample random ids and compare token strings
    gen = torch.Generator().manual_seed(0)
    ids = torch.randint(low=0, high=tok_a.vocab_size, size=(sample,), generator=gen).tolist()
    a_tokens = tok_a.convert_ids_to_tokens(ids)
    b_tokens = tok_b.convert_ids_to_tokens(ids)
    return a_tokens == b_tokens


@dataclass
class ModelBundle:
    name: str
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    device: torch.device


def load_model(
    model_name: str,
    device_str: str,
    dtype_str: str = "bfloat16",
    trust_remote_code: bool = False,
) -> ModelBundle:
    if not torch.cuda.is_available() and device_str.startswith("cuda"):
        raise RuntimeError("CUDA requested but not available.")

    if dtype_str.lower() == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype_str.lower() in ("float16", "fp16"):
        torch_dtype = torch.float16
    elif dtype_str.lower() in ("float32", "fp32"):
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unknown dtype: {dtype_str}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        # Ensure a PAD token exists for batching; reuse EOS as PAD if needed
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device(device_str if device_str != "cpu" else "cpu")

    # Put the entire model on the specified device
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
        device_map={"": device.index if device.type == "cuda" else device_str},
    )
    model.eval()
    return ModelBundle(name=model_name, tokenizer=tokenizer, model=model, device=device)


def js_divergence_from_logs(logp_a: torch.Tensor, logp_b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    logp_a, logp_b: [N, V] log-probabilities
    JS = 0.5 * KL(P || M) + 0.5 * KL(Q || M), where M = 0.5*(P+Q).
    Compute with F.kl_div using log_target=True for numeric stability.
    Returns: [N] per-row JS divergence.
    """
    # p, q implicitly via log-probs; build log_m via logsumexp trick
    # m = 0.5*(exp(logp_a) + exp(logp_b))
    # log_m = log(0.5*exp(logp_a) + 0.5*exp(logp_b))
    #      = logsumexp([logp_a + log(0.5), logp_b + log(0.5)], dim=-1)
    log_half = math.log(0.5)
    stacked = torch.stack([logp_a + log_half, logp_b + log_half], dim=0)  # [2, N, V]
    log_m = torch.logsumexp(stacked, dim=0)  # [N, V]

    # KL(P || M) with log_target=True means F.kl_div(log_m, logp_a, log_target=True)
    # returns sum(exp(logp_a) * (logp_a - log_m)) per row.
    kl_p_m = F.kl_div(log_m, logp_a, reduction="none", log_target=True).sum(-1)
    kl_q_m = F.kl_div(log_m, logp_b, reduction="none", log_target=True).sum(-1)
    return 0.5 * (kl_p_m + kl_q_m)


def kl_forward_reverse_from_logs(
    logp_a: torch.Tensor, logp_b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-row forward KL D_KL(A||B) and reverse KL D_KL(B||A) from log-probs.
    Using log_target=True to avoid materializing probabilities.
    Returns: (kl_fwd, kl_rev), each [N].
    """
    # D_KL(A || B): target = logp_a, input = logp_b
    kl_fwd = F.kl_div(logp_b, logp_a, reduction="none", log_target=True).sum(-1)
    # D_KL(B || A)
    kl_rev = F.kl_div(logp_a, logp_b, reduction="none", log_target=True).sum(-1)
    return kl_fwd, kl_rev


def chunked_log_softmax(logits: torch.Tensor, chunk: int = 128) -> torch.Tensor:
    """
    Compute log_softmax over vocab in manageable T-chunks.
    logits: [N, V] (already flattened over batch/time) float32 for stability.
    Returns log-probs of same shape.
    """
    N, V = logits.shape
    out = torch.empty_like(logits, dtype=torch.float32)
    for i in range(0, N, chunk):
        sl = slice(i, min(i + chunk, N))
        out[sl] = F.log_softmax(logits[sl], dim=-1)
    return out


# -------------------------------
# Core evaluation loop
# -------------------------------

@torch.no_grad()
def evaluate_kl(
    ds,
    mdl_a: ModelBundle,
    mdl_b: ModelBundle,
    batch_size: int = 1,
    num_workers: int = 0,
    time_chunk: int = 128,
    max_batches: Optional[int] = None,
    save_csv: Optional[str] = None,
):
    """
    Runs teacher-forcing through both models and computes KL metrics only on response tokens,
    identified by the dataset's loss_mask (shifted like standard next-token loss).
    """

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    # loader = ds 
    total_tok = 0
    total_kl_fwd = 0.0
    total_kl_rev = 0.0
    total_js = 0.0

    per_example_rows: List[dict] = []
    seen_batches = 0

    for batch_idx, batch in enumerate(loader):
        # Stop early (useful for dev)
        if max_batches is not None and seen_batches >= max_batches:
            break
        seen_batches += 1 

        # print(batch)
        # print("batch above ------------------------------")

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        # position_ids = batch["position_ids"]
        # loss_mask = batch.pop("loss_mask")[:, :-1].reshape(-1)
        # We'll rely on attention_mask only; position_ids not required for most HF models.

        # Move batch once per model
        ids_a = input_ids.to(mdl_a.device, non_blocking=True)
        am_a = attention_mask.to(mdl_a.device, non_blocking=True)
        ids_b = input_ids.to(mdl_b.device, non_blocking=True)
        am_b = attention_mask.to(mdl_b.device, non_blocking=True)
        
        # Forward both models
        out_a = mdl_a.model(input_ids=ids_a, attention_mask=am_a, use_cache=False)
        out_b = mdl_b.model(input_ids=ids_b, attention_mask=am_b, use_cache=False)

        # Use float32 for stable log-softmax/kl
        logits_a = out_a.logits.float().to("cpu")
        logits_b = out_b.logits.float().to("cpu")

        B, S, V = logits_a.shape
        assert logits_b.shape == (B, S, V), "Logits shapes must match (tokenizers mismatch?)"

        # Shift to next-token prediction
        T = S - 1
        shift_logits_a = logits_a[:, :-1, :]  # [B, T, V]
        shift_logits_b = logits_b[:, :-1, :]  # [B, T, V]

        # Mask of valid response tokens (the same mask used in your SFT loss)
        loss_mask = batch["loss_mask"][:, :-1].to("cpu").bool()  # [B, T]

        # Flatten time to compute in chunks
        flat_a = shift_logits_a.reshape(-1, V)  # [B*T, V]
        flat_b = shift_logits_b.reshape(-1, V)
        flat_mask = loss_mask.reshape(-1)      # [B*T]

        # Compute log-probs in chunks (over vocab)
        logp_a = chunked_log_softmax(flat_a, chunk=time_chunk)  # [N, V]
        logp_b = chunked_log_softmax(flat_b, chunk=time_chunk)

        # Per-position divergences
        kl_fwd_per = torch.empty(logp_a.size(0), dtype=torch.float32)
        kl_rev_per = torch.empty_like(kl_fwd_per)
        js_per = torch.empty_like(kl_fwd_per)

        # Chunk over rows to limit temp memory inside kl_div
        row_chunk = max(1, 8192 // max(1, V // 4096))  # heuristic
        for i in range(0, logp_a.size(0), row_chunk):
            sl = slice(i, min(i + row_chunk, logp_a.size(0)))
            kf, kr = kl_forward_reverse_from_logs(logp_a[sl], logp_b[sl])
            j = js_divergence_from_logs(logp_a[sl], logp_b[sl])
            kl_fwd_per[sl] = kf
            kl_rev_per[sl] = kr
            js_per[sl] = j

        # Keep only response tokens
        sel = flat_mask
        num_tok = int(sel.sum().item())
        if num_tok == 0:
            continue

        sum_kf = kl_fwd_per[sel].sum().item()
        sum_kr = kl_rev_per[sel].sum().item()
        sum_js = js_per[sel].sum().item()

        total_tok += num_tok
        total_kl_fwd += sum_kf
        total_kl_rev += sum_kr
        total_js += sum_js

        # Per-example metrics (averaged over response tokens of that example)
        kf_bt = kl_fwd_per.view(B, T)
        kr_bt = kl_rev_per.view(B, T)
        js_bt = js_per.view(B, T)
        mask_bt = loss_mask  # [B, T]

        tok_per_ex = mask_bt.sum(dim=1).to(torch.int64)  # [B]
        sum_kf_ex = (kf_bt * mask_bt).sum(dim=1)
        sum_kr_ex = (kr_bt * mask_bt).sum(dim=1)
        sum_js_ex = (js_bt * mask_bt).sum(dim=1)

        for i_ex in range(B):
            tcount = int(tok_per_ex[i_ex].item())
            if tcount == 0:
                continue
            per_example_rows.append({
                "batch_index": batch_idx,
                "example_index_in_batch": i_ex,
                "response_tokens": tcount,
                "kl_fwd_avg": float(sum_kf_ex[i_ex].item() / tcount),
                "kl_rev_avg": float(sum_kr_ex[i_ex].item() / tcount),
                "js_avg": float(sum_js_ex[i_ex].item() / tcount),
            })

    # Aggregate
    avg_kf = total_kl_fwd / max(1, total_tok)
    avg_kr = total_kl_rev / max(1, total_tok)
    avg_js = total_js / max(1, total_tok)

    # Save CSV if requested
    if save_csv:
        import pandas as pd
        df = pd.DataFrame(per_example_rows)
        df.to_csv(save_csv, index=False)
        print(f"[INFO] Wrote per-example metrics to {save_csv}")

    print("\n========== KL RESULTS (response tokens only) ==========")
    print(f"Total masked tokens: {total_tok:,}")
    print(f"Forward KL  D_KL(A || B): {avg_kf:.6f}  (avg per token)")
    print(f"Reverse KL  D_KL(B || A): {avg_kr:.6f}  (avg per token)")
    print(f"JS Divergence         : {avg_js:.6f}  (avg per token)")
    print("=======================================================\n")


# -------------------------------
# Main (args & dataset build)
# -------------------------------

def build_config(args) -> dict:
    """
    Mirror your provided config shape; you can tweak here if needed.
    """
    return {'train_batch_size': 256, 'micro_batch_size': None, 'micro_batch_size_per_gpu': 4, 'train_files': args.parquet, 'val_files': 'data/math500/test.parquet', 'prompt_key': 'extra_info', 'response_key': 'extra_info', 'prompt_dict_keys': ['question'], 'response_dict_keys': ['answer'], 'multiturn': {'enable': False, 'messages_key': 'messages', 'tools_key': 'tools', 'enable_thinking_key': 'enable_thinking'}, 'max_length': 4096, 'truncation': 'right', 'balance_dp_token': False, 'chat_template': None, 'custom_cls': {'path': None, 'name': None}, 'use_shm': False}


def parse_args():
    p = argparse.ArgumentParser(description="Measure KL divergence between two causal LMs on SFT responses.")
    p.add_argument("--parquet", type=str, default="/shared/nas2/xiusic/Benchmarks/selective_FT/DFT/verl/data/numina_cot_PRM/train.parquet", help="Path to parquet file (or comma-separated list).")
    p.add_argument("--model_a", type=str, default="Qwen/Qwen2.5-Math-1.5B", help="HF model name/path for Model A (P in D_KL(P||Q)).")
    p.add_argument("--model_b", type=str, default="/shared/nas2/xiusic/Benchmarks/selective_FT/DFT/verl/checkpoints/numina-cot-qwen-2.5-math-1.5b-lr-5e-5-bz-256-max_length-3072-nproc_per_node-2-micro_batch_size-1-0813-67k-samples-dft/global_step_264", help="HF model name/path for Model B (Q in D_KL(P||Q)).")
    # p.add_argument("--model_b", type=str, default="/shared/nas2/xiusic/Benchmarks/selective_FT/DFT/verl/checkpoints/numina-cot-qwen-2.5-math-1.5b-lr-5e-5-bz-256-max_length-3072-nproc_per_node-1-micro_batch_size-1-0813-67k-samples-sft/global_step_264", help="HF model name/path for Model B (Q in D_KL(P||Q)).")

    p.add_argument("--device_a", type=str, default="cuda:0", help="Device for Model A (e.g., cuda:0, cuda:1, cpu).")
    p.add_argument("--device_b", type=str, default="cuda:1", help="Device for Model B.")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Model weights dtype.")
    p.add_argument("--trust_remote_code", action="store_true", help="Pass through to HF loaders.")

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--time_chunk", type=int, default=128, help="Chunk size over time (rows) when computing log_softmax/kl.")
    p.add_argument("--max_batches", type=int, default=10, help="For quick tests; limit number of batches processed.")

    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--truncation", type=str, default="right", choices=["right", "left"])

    p.add_argument("--save_csv", type=str, default="/shared/nas2/xiusic/Benchmarks/selective_FT/KL_logs/save_log.csv", help="Optional path to save per-example metrics CSV.")
    return p.parse_args()


def main():
    args = parse_args()

    # Load models
    print(f"[INFO] Loading Model A on {args.device_a}: {args.model_a}")
    mdl_a = load_model(args.model_a, device_str=args.device_a, dtype_str=args.dtype, trust_remote_code=args.trust_remote_code)
    print(f"[INFO] Loading Model B on {args.device_b}: {args.model_b}")
    mdl_b = load_model(args.model_b, device_str=args.device_b, dtype_str=args.dtype, trust_remote_code=args.trust_remote_code)

    # Enforce same tokenizer/vocab
    print("[INFO] Checking tokenizer compatibility...")
    if not tokenizers_compatible(mdl_a.tokenizer, mdl_b.tokenizer):
        raise RuntimeError(
            "Tokenizers are not compatible. KL over token distributions requires identical vocab/token->id mapping.\n"
            "Choose models from the same family (e.g., two LLaMA variants) or unify tokenizers."
        )
    tokenizer = mdl_a.tokenizer  # use a single tokenizer for the dataset

    # Build dataset using your class + config
    parquet_files = args.parquet
    cfg = build_config(args)
    ds = SFTDataset_PRM(parquet_files, tokenizer, cfg)

    # print(ds[0])
    # exit()

    print(f"[INFO] Dataset size: {len(ds)} examples")
    print(f"[INFO] Vocab size: {tokenizer.vocab_size}")

    evaluate_kl(
        ds=ds,
        mdl_a=mdl_a,
        mdl_b=mdl_b,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        time_chunk=args.time_chunk,
        max_batches=args.max_batches,
        save_csv=args.save_csv,
    )


if __name__ == "__main__":
    main()
