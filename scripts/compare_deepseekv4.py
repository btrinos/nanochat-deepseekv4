"""
Compare native nanochat GPT against the scaled DeepSeek-V4 nanochat variant.

Examples:
    python -m scripts.compare_deepseekv4 --steps 40 --device mps
    python -m scripts.compare_deepseekv4 --datasets tiny_shakespeare,wikitext2 --steps 80
"""

import argparse
import csv
import json
import math
import os
import socket
import time
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F

from nanochat.common import autodetect_device_type
from nanochat.gpt import GPT, GPTConfig
from nanochat.deepseek_v4 import DeepSeekV4NanoChat, DeepSeekV4NanoConfig


TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
WIKITEXT_PARQUET_URL = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1"
VOCAB_SIZE = 257


def fetch_url(url, timeout=120, retries=5):
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except (TimeoutError, socket.timeout, urllib.error.URLError):
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 20))


def read_or_fetch_text(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(fetch_url(url))
    return path.read_text(encoding="utf-8", errors="replace")


def load_tiny_shakespeare(cache_dir):
    text = read_or_fetch_text(TINY_SHAKESPEARE_URL, cache_dir / "tiny_shakespeare.txt")
    n = len(text)
    train_end = int(0.9 * n)
    val_end = int(0.95 * n)
    return {
        "train": text[:train_end],
        "validation": text[train_end:val_end],
        "test": text[val_end:],
    }


def fetch_wikitext_split(split, cache_dir, max_chars=None):
    suffix = "full" if max_chars is None else str(max_chars)
    path = cache_dir / f"wikitext2_{split}_{suffix}.txt"
    tmp_path = cache_dir / f"wikitext2_{split}_{suffix}.partial.json"
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")

    if tmp_path.exists():
        state = json.loads(tmp_path.read_text())
        texts = state["texts"]
        chars = state["chars"]
        offset = state["offset"]
    else:
        texts = []
        chars = 0
        offset = 0
    page_len = 100
    while True:
        params = urllib.parse.urlencode({
            "dataset": "Salesforce/wikitext",
            "config": "wikitext-2-raw-v1",
            "split": split,
            "offset": offset,
            "length": page_len,
        })
        payload = json.loads(fetch_url(f"{HF_ROWS_URL}?{params}").decode("utf-8"))
        rows = payload.get("rows", [])
        if not rows:
            break
        for row in rows:
            text = row["row"].get("text", "")
            if text.strip():
                texts.append(text)
                chars += len(text) + 1
                if max_chars is not None and chars >= max_chars:
                    break
        if max_chars is not None and chars >= max_chars:
            break
        if len(rows) < page_len:
            break
        offset += page_len
        tmp_path.write_text(json.dumps({"texts": texts, "chars": chars, "offset": offset}))

    text = "\n".join(texts)
    if max_chars is not None:
        text = text[:max_chars]
    path.write_text(text, encoding="utf-8")
    if tmp_path.exists():
        tmp_path.unlink()
    return text


def fetch_wikitext_split_parquet(split, cache_dir):
    text_path = cache_dir / f"wikitext2_{split}_full.txt"
    if text_path.exists():
        return text_path.read_text(encoding="utf-8", errors="replace")

    import pyarrow.parquet as pq

    parquet_path = cache_dir / f"wikitext2_{split}.parquet"
    if not parquet_path.exists():
        parquet_path.write_bytes(fetch_url(f"{WIKITEXT_PARQUET_URL}/{split}-00000-of-00001.parquet", timeout=300))
    table = pq.read_table(parquet_path, columns=["text"])
    texts = [text for text in table.column("text").to_pylist() if text.strip()]
    text = "\n".join(texts)
    text_path.write_text(text, encoding="utf-8")
    return text


def load_wikitext2(cache_dir, train_chars, eval_chars, full_data=False):
    if full_data:
        return {
            "train": fetch_wikitext_split_parquet("train", cache_dir),
            "validation": fetch_wikitext_split_parquet("validation", cache_dir),
            "test": fetch_wikitext_split_parquet("test", cache_dir),
        }
    return {
        "train": fetch_wikitext_split("train", cache_dir, train_chars),
        "validation": fetch_wikitext_split("validation", cache_dir, eval_chars),
        "test": fetch_wikitext_split("test", cache_dir, eval_chars),
    }


def encode_bytes(text):
    return torch.tensor(list(text.encode("utf-8", errors="replace")), dtype=torch.long)


def get_batch(tokens, batch_size, seq_len, device, rng, positions=None):
    max_start = len(tokens) - seq_len - 1
    if max_start <= 0:
        raise ValueError(f"Dataset split is too short for seq_len={seq_len}: {len(tokens)} byte tokens")
    if positions is None:
        positions = torch.arange(seq_len + 1, device=device)
    starts = torch.randint(0, max_start, (batch_size,), generator=rng, device="cpu").to(device)
    offsets = starts[:, None] + positions[None, :]
    batch = tokens.index_select(0, offsets.reshape(-1)).view(batch_size, seq_len + 1)
    return batch[:, :-1], batch[:, 1:]


def build_native(seq_len, n_layer, n_embd, n_head):
    config = GPTConfig(
        sequence_len=seq_len,
        vocab_size=VOCAB_SIZE,
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=n_head,
        n_embd=n_embd,
        window_pattern="L",
    )
    model = GPT(config, pad_vocab_size_to=64)
    model.init_weights()
    return model


def build_deepseekv4(seq_len, n_layer, n_embd, n_head):
    config = DeepSeekV4NanoConfig(
        sequence_len=seq_len,
        vocab_size=VOCAB_SIZE,
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=1,
        n_embd=n_embd,
        window_size=max(16, seq_len // 2),
        attention_layer_pattern="CSA,HCA",
        csa_compress_ratio=4,
        hca_compress_ratio=16,
        rope_head_dim=max(8, (n_embd // n_head) // 2),
        q_lora_rank=max(16, n_embd // 2),
        o_lora_rank=max(16, n_embd // 2),
        o_groups=min(n_head, 4),
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=max(64, n_embd),
        n_hash_layers=1 if n_layer > 1 else 0,
        routed_scaling_factor=1.0,
        aux_free_balance_rate=1e-3,
        sequence_balance_loss_weight=1e-2,
        hc_mult=2,
        hc_sinkhorn_iters=4,
        num_nextn_predict_layers=1,
        mtp_loss_weight=0.1,
        original_max_position_embeddings=seq_len,
        index_topk=4,
    )
    model = DeepSeekV4NanoChat(config, pad_vocab_size_to=64)
    model.init_weights()
    return model


def primary_loss(model, x, y):
    if isinstance(model, DeepSeekV4NanoChat):
        logits = model(x)
    else:
        logits = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()


@torch.inference_mode()
def evaluate(model, tokens, batch_size, seq_len, device, eval_batches, seed):
    model.eval()
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    positions = torch.arange(seq_len + 1, device=device)
    losses = []
    for _ in range(eval_batches):
        x, y = get_batch(tokens, batch_size, seq_len, device, rng, positions)
        losses.append(primary_loss(model, x, y))
    model.train()
    return sum(losses) / len(losses)


@torch.inference_mode()
def evaluate_full_split(model, tokens, batch_size, seq_len, device):
    model.eval()
    max_start = len(tokens) - seq_len - 1
    starts = torch.arange(0, max_start, seq_len, device=device)
    positions = torch.arange(seq_len + 1, device=device)
    total_loss = torch.zeros((), device=device)
    total_tokens = 0
    for i in range(0, starts.numel(), batch_size):
        batch_starts = starts[i:i + batch_size]
        offsets = batch_starts[:, None] + positions[None, :]
        batch = tokens.index_select(0, offsets.reshape(-1)).view(batch_starts.numel(), seq_len + 1)
        x = batch[:, :-1]
        y = batch[:, 1:]
        if isinstance(model, DeepSeekV4NanoChat):
            logits = model(x)
        else:
            logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_loss = total_loss + loss
        total_tokens += y.numel()
    model.train()
    return float((total_loss / total_tokens).detach().cpu())


def make_fixed_starts(num_tokens, seq_len, count, seed):
    max_start = num_tokens - seq_len - 1
    if max_start <= 0:
        raise ValueError(f"Dataset split is too short for seq_len={seq_len}: {num_tokens} byte tokens")
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    return torch.randint(0, max_start, (count,), generator=rng, device="cpu")


@torch.inference_mode()
def evaluate_fixed_starts(model, tokens, starts, batch_size, seq_len, device):
    model.eval()
    positions = torch.arange(seq_len + 1, device=device)
    total_loss = torch.zeros((), device=device)
    total_tokens = 0
    for i in range(0, starts.numel(), batch_size):
        batch_starts = starts[i:i + batch_size].to(device)
        offsets = batch_starts[:, None] + positions[None, :]
        batch = tokens.index_select(0, offsets.reshape(-1)).view(batch_starts.numel(), seq_len + 1)
        x = batch[:, :-1]
        y = batch[:, 1:]
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_loss = total_loss + loss
        total_tokens += y.numel()
    model.train()
    return float((total_loss / total_tokens).detach().cpu())


def estimate_active_params(model):
    total = sum(p.numel() for p in model.parameters())
    if not isinstance(model, DeepSeekV4NanoChat):
        return total
    routed_total = 0
    for block in model.transformer.h:
        routed_total += sum(p.numel() for expert in block.moe.experts for p in expert.parameters())
    mtp_total = sum(p.numel() for p in model.mtp_heads.parameters())
    active_routed = round(routed_total * model.config.num_experts_per_tok / max(1, model.config.n_routed_experts))
    return total - routed_total - mtp_total + active_routed


def perplexity(loss):
    return math.exp(min(loss, 20.0))


def bits_per_byte(loss):
    return loss / math.log(2.0)


def display_dataset_name(dataset):
    if dataset == "wikitext2":
        return "WikiText-2"
    if dataset == "tiny_shakespeare":
        return "Tiny Shakespeare"
    return dataset.replace("_", " ").title()


def train_one(model_name, model, splits, args, device):
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_tokens = encode_bytes(splits["train"]).to(device)
    val_tokens = encode_bytes(splits["validation"]).to(device)
    test_tokens = encode_bytes(splits["test"]).to(device)

    rng = torch.Generator(device="cpu")
    rng.manual_seed(args.seed)
    positions = torch.arange(args.seq_len + 1, device=device)
    eval_batch_size = args.eval_batch_size or args.batch_size
    train_eval_starts = make_fixed_starts(
        train_tokens.numel(),
        args.seq_len,
        args.train_eval_batches * eval_batch_size,
        args.seed + 3000,
    )
    params = sum(p.numel() for p in model.parameters())
    active_params = estimate_active_params(model)
    records = []
    start_time = time.time()
    for step in range(1, args.steps + 1):
        x, y = get_batch(train_tokens, args.batch_size, args.seq_len, device, rng, positions)
        optimizer.zero_grad(set_to_none=True)
        if isinstance(model, DeepSeekV4NanoChat):
            loss = model(x, y, include_aux_loss=True)
        else:
            loss = model(x, y)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if isinstance(model, DeepSeekV4NanoChat):
            model.update_aux_free_balance()

        should_eval = (
            (step == 1 and not args.skip_initial_eval)
            or (args.eval_every > 0 and step % args.eval_every == 0)
            or step == args.steps
        )
        if should_eval:
            train_loss = evaluate_fixed_starts(
                model, train_tokens, train_eval_starts, eval_batch_size, args.seq_len, device
            )
            if args.full_eval:
                val_loss = evaluate_full_split(model, val_tokens, eval_batch_size, args.seq_len, device)
                test_loss = evaluate_full_split(model, test_tokens, eval_batch_size, args.seq_len, device)
            else:
                val_loss = evaluate(model, val_tokens, args.batch_size, args.seq_len, device, args.eval_batches, args.seed + 1000 + step)
                test_loss = evaluate(model, test_tokens, args.batch_size, args.seq_len, device, args.eval_batches, args.seed + 2000 + step)
            elapsed = time.time() - start_time
            records.append({
                "model": model_name,
                "step": step,
                "tokens_seen": step * args.batch_size * args.seq_len,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "test_loss": test_loss,
                "train_objective": float(loss.detach().cpu()),
                "seconds": elapsed,
                "params": params,
                "active_params": active_params,
            })
            print(
                f"{model_name:22s} step {step:04d}/{args.steps:04d} "
                f"train {train_loss:.4f} val {val_loss:.4f} test {test_loss:.4f} "
                f"train_obj {float(loss.detach().cpu()):.4f}",
                flush=True,
            )
    return records


def format_token_tick(value, _pos=None):
    if abs(value) >= 1000:
        return f"{value / 1000:.0f}k"
    return f"{value:.0f}"


def set_loss_axis(ax, values):
    from matplotlib.ticker import MultipleLocator

    if not values:
        return
    y_min = min(values)
    y_max = max(values)
    spread = y_max - y_min
    padding = max(spread * 0.12, 0.02)
    ax.set_ylim(y_min - padding, y_max + padding)
    if spread <= 0.35:
        major, minor = 0.05, 0.01
    elif spread <= 0.8:
        major, minor = 0.10, 0.02
    else:
        major, minor = 0.25, 0.05
    ax.yaxis.set_major_locator(MultipleLocator(major))
    ax.yaxis.set_minor_locator(MultipleLocator(minor))


def style_token_axis(ax, max_tokens):
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    ax.set_xlim(0, max_tokens)
    ax.xaxis.set_major_locator(MultipleLocator(10000))
    ax.xaxis.set_minor_locator(MultipleLocator(5000))
    ax.xaxis.set_major_formatter(FuncFormatter(format_token_tick))


def plot_results(records, output_path, metric_titles=None, y_label="cross-entropy loss"):
    import matplotlib.pyplot as plt

    datasets = sorted({r["dataset"] for r in records})
    fig, axes = plt.subplots(len(datasets), 2, figsize=(11.6, 3.7 * len(datasets)), squeeze=False)
    colors = {"native-nanochat": "#2f6fa3", "nanochat-deepseekv4": "#c44e36"}
    if metric_titles is None:
        metric_titles = [("train_loss", "Train eval loss"), ("validation_loss", "Validation loss")]
    for row_idx, dataset in enumerate(datasets):
        subset = [r for r in records if r["dataset"] == dataset]
        display_name = display_dataset_name(dataset)
        for col_idx, (metric, title) in enumerate(metric_titles):
            ax = axes[row_idx, col_idx]
            dataset_tokens = []
            for model_name in ["native-nanochat", "nanochat-deepseekv4"]:
                rows = [r for r in subset if r["model"] == model_name]
                if not rows:
                    continue
                tokens_seen = [r["tokens_seen"] for r in rows]
                dataset_tokens.extend(tokens_seen)
                ax.plot(
                    tokens_seen,
                    [r[metric] for r in rows],
                    label=model_name,
                    color=colors[model_name],
                    linewidth=1.7,
                    marker="o",
                    markersize=6.5,
                )
            metric_values = [r[metric] for r in subset]
            set_loss_axis(ax, metric_values)
            ax.set_title(f"{display_name}: {title}", fontsize=15)
            if dataset_tokens:
                style_token_axis(ax, max(dataset_tokens))
            ax.set_xlabel("training tokens seen")
            ax.set_ylabel(y_label)
            ax.grid(True, which="major", alpha=0.32)
            ax.grid(True, which="minor", alpha=0.12)
            ax.legend(loc="upper right", frameon=False)
    fig.suptitle("Train eval and validation loss: native nanochat vs nanochat-deepseekv4", fontsize=16)
    fig.tight_layout(h_pad=2.0, rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_validation_curves(records, output_path):
    import matplotlib.pyplot as plt

    datasets = sorted({r["dataset"] for r in records})
    fig, axes = plt.subplots(1, len(datasets), figsize=(11.6, 4.0), squeeze=False)
    colors = {"native-nanochat": "#2f6fa3", "nanochat-deepseekv4": "#c44e36"}
    for col_idx, dataset in enumerate(datasets):
        ax = axes[0, col_idx]
        subset = [r for r in records if r["dataset"] == dataset]
        dataset_tokens = []
        for model_name in ["native-nanochat", "nanochat-deepseekv4"]:
            rows = [r for r in subset if r["model"] == model_name]
            if not rows:
                continue
            tokens_seen = [r["tokens_seen"] for r in rows]
            dataset_tokens.extend(tokens_seen)
            ax.plot(
                tokens_seen,
                [r["validation_loss"] for r in rows],
                label=model_name,
                color=colors[model_name],
                linewidth=1.8,
                marker="o",
                markersize=6.5,
            )
        set_loss_axis(ax, [r["validation_loss"] for r in subset])
        if dataset_tokens:
            style_token_axis(ax, max(dataset_tokens))
        ax.set_title(display_dataset_name(dataset), fontsize=15)
        ax.set_xlabel("training tokens seen")
        ax.set_ylabel("validation cross-entropy")
        ax.grid(True, which="major", alpha=0.32)
        ax.grid(True, which="minor", alpha=0.12)
        ax.legend(loc="upper right", frameon=False)
    fig.suptitle("Validation loss: native nanochat vs nanochat-deepseekv4", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def final_records(records):
    finals = []
    datasets = sorted({r["dataset"] for r in records})
    for dataset in datasets:
        for model_name in ["native-nanochat", "nanochat-deepseekv4"]:
            rows = [r for r in records if r["dataset"] == dataset and r["model"] == model_name]
            if rows:
                finals.append(rows[-1])
    return finals


def plot_final_bars(records, output_path):
    import matplotlib.pyplot as plt

    finals = final_records(records)
    datasets = sorted({r["dataset"] for r in finals})
    fig, axes = plt.subplots(len(datasets), 2, figsize=(11.6, 3.8 * len(datasets)), squeeze=False)
    colors = {"native-nanochat": "#2f6fa3", "nanochat-deepseekv4": "#c44e36"}
    for row_idx, dataset in enumerate(datasets):
        by_model = {r["model"]: r for r in finals if r["dataset"] == dataset}
        for col_idx, (key, title) in enumerate([
            ("test_loss", "Test loss"),
            ("validation_loss", "Validation loss"),
        ]):
            ax = axes[row_idx, col_idx]
            model_names = ["native-nanochat", "nanochat-deepseekv4"]
            values = [by_model[model_name][key] for model_name in model_names]
            bars = ax.bar(
                range(len(values)),
                values,
                color=[colors[model_name] for model_name in model_names],
                width=0.58,
            )
            ax.set_xticks(range(len(values)), ["native-nanochat", "nanochat-deepseekv4"])
            ax.set_title(f"{display_dataset_name(dataset)}: {title}", fontsize=15)
            ax.set_ylabel("cross-entropy loss")
            ax.grid(True, axis="y", alpha=0.25)
            y_min = min(values)
            y_max = max(values)
            padding = max((y_max - y_min) * 0.18, y_max * 0.02)
            ax.set_ylim(max(0, y_min - padding), y_max + padding)
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
    fig.suptitle("Final held-out loss: native nanochat vs nanochat-deepseekv4", fontsize=16)
    fig.tight_layout(h_pad=2.0, rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_csv(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "model",
        "step",
        "tokens_seen",
        "train_loss",
        "validation_loss",
        "test_loss",
        "train_objective",
        "seconds",
        "params",
        "active_params",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def write_summary_csv(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "model",
        "split",
        "step",
        "tokens_seen",
        "loss",
        "perplexity",
        "bits_per_byte",
        "params",
        "active_params",
    ]
    rows = []
    for row in final_records(records):
        for split, key in [
            ("train_eval", "train_loss"),
            ("validation", "validation_loss"),
            ("test", "test_loss"),
        ]:
            loss = row[key]
            rows.append({
                "dataset": row["dataset"],
                "model": row["model"],
                "split": split,
                "step": row["step"],
                "tokens_seen": row["tokens_seen"],
                "loss": loss,
                "perplexity": perplexity(loss),
                "bits_per_byte": bits_per_byte(loss),
                "params": row["params"],
                "active_params": row["active_params"],
            })
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="tiny_shakespeare,wikitext2", help="comma-separated: tiny_shakespeare,wikitext2")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=256, help="batch size for --full-eval; 0 uses --batch-size")
    parser.add_argument("--train-eval-batches", type=int, default=32, help="fixed train-eval batches for pure next-token cross-entropy")
    parser.add_argument("--full-data", action="store_true", help="use full WikiText-2 train/validation/test splits instead of character caps")
    parser.add_argument("--full-eval", action="store_true", help="evaluate every non-overlapping window in validation/test splits")
    parser.add_argument("--skip-initial-eval", action="store_true", help="do not evaluate after step 1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--train-chars", type=int, default=600_000)
    parser.add_argument("--eval-chars", type=int, default=120_000)
    parser.add_argument("--output-dir", default="artifacts/deepseekv4_compare")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device_type = autodetect_device_type() if args.device == "auto" else args.device
    device = torch.device(device_type)
    cache_dir = Path(args.output_dir) / "data_cache"

    loaders = {
        "tiny_shakespeare": lambda: load_tiny_shakespeare(cache_dir),
        "wikitext2": lambda: load_wikitext2(cache_dir, args.train_chars, args.eval_chars, full_data=args.full_data),
    }
    selected = [name.strip() for name in args.datasets.split(",") if name.strip()]
    all_records = []
    for dataset_name in selected:
        if dataset_name not in loaders:
            raise ValueError(f"Unknown dataset {dataset_name}. Choose from {sorted(loaders)}")
        print(f"\n=== Dataset: {dataset_name} ===")
        splits = loaders[dataset_name]()

        for model_name, builder in [
            ("native-nanochat", build_native),
            ("nanochat-deepseekv4", build_deepseekv4),
        ]:
            torch.manual_seed(args.seed)
            model = builder(args.seq_len, args.n_layer, args.n_embd, args.n_head)
            params = sum(p.numel() for p in model.parameters())
            print(f"{model_name}: {params:,} parameters")
            records = train_one(model_name, model, splits, args, device)
            for row in records:
                row["dataset"] = dataset_name
            all_records.extend(records)
            del model
            if device_type == "mps":
                torch.mps.empty_cache()
            elif device_type == "cuda":
                torch.cuda.empty_cache()

    output_dir = Path(args.output_dir)
    csv_path = output_dir / "losses.csv"
    summary_csv_path = output_dir / "summary.csv"
    validation_png_path = output_dir / f"validation_loss_{args.steps}_steps.png"
    train_val_png_path = output_dir / f"train_validation_ce_{args.steps}_steps.png"
    final_bar_png_path = output_dir / f"final_validation_test_bars_{args.steps}_steps.png"
    write_csv(all_records, csv_path)
    write_summary_csv(all_records, summary_csv_path)
    plot_validation_curves(all_records, validation_png_path)
    plot_results(
        all_records,
        train_val_png_path,
        metric_titles=[("train_loss", "Train eval loss"), ("validation_loss", "Validation loss")],
        y_label="cross-entropy loss",
    )
    plot_final_bars(all_records, final_bar_png_path)

    print("\nFinal losses")
    for dataset_name in selected:
        for model_name in ["native-nanochat", "nanochat-deepseekv4"]:
            rows = [r for r in all_records if r["dataset"] == dataset_name and r["model"] == model_name]
            row = rows[-1]
            print(
                f"{dataset_name:18s} {model_name:22s} "
                f"train {row['train_loss']:.4f} val {row['validation_loss']:.4f} "
                f"test {row['test_loss']:.4f} params {row['params']:,} "
                f"active {row['active_params']:,}"
            )
    print(f"\nWrote {csv_path}")
    print(f"Wrote {summary_csv_path}")
    print(f"Wrote {validation_png_path}")
    print(f"Wrote {train_val_png_path}")
    print(f"Wrote {final_bar_png_path}")


if __name__ == "__main__":
    main()
