# eval.py
import os
import math
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import tiktoken

from datasets import load_dataset
from transformers import GPT2Config, GPT2LMHeadModel

from modeling import convert_gpt2_to_moe
from patches import patch_model_for_gmoe
from utils import (
    set_seed,
    load_safetensors,
    chunked_cross_entropy,
    enable_sdp_backends,
    prefer_flash_attention,
    print_attn_stack_status,
)

from data import (
    load_pile_test,
    worker_init_fn,
    get_dataloader_generator,
)

from config import (
    CHECKPOINTS_DIR,
    MODEL_SPECS,
    HF_HOME,
    HF_DATASETS_CACHE,
)


os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_DATASETS_CACHE", HF_DATASETS_CACHE)

enc = tiktoken.get_encoding("gpt2")

EVAL_DATASETS = [
    "pile",
    "wikitext103",
    "cc100",
    "openwebtext",
]


def load_wikitext103():
    cache_dir = os.environ.get("HF_DATASETS_CACHE", None)
    dataset = load_dataset(
        "Geonwoohong/wikitext-103-raw-v1-test-tokenized-gpt2",
        cache_dir=cache_dir,
    )
    return dataset["test"]


def load_cc100():
    cache_dir = os.environ.get("HF_DATASETS_CACHE", None)
    dataset = load_dataset(
        "Geonwoohong/cc100-en-test-tokenized-gpt2",
        cache_dir=cache_dir,
    )
    return dataset["test"]


def load_openwebtext():
    cache_dir = os.environ.get("HF_DATASETS_CACHE", None)
    dataset = load_dataset(
        "Geonwoohong/openwebtext-test-tokenized-gpt2",
        cache_dir=cache_dir,
    )
    return dataset["test"]


def load_eval_dataset(dataset_name: str, pile_subset_ratio: float = 0.01):
    dataset_name = dataset_name.lower()

    if dataset_name == "pile":
        dataset = load_pile_test(verbose=True)
        subset_size = max(1, int(len(dataset) * pile_subset_ratio))
        dataset = dataset.select(range(subset_size))
        display_name = f"Pile Test ({pile_subset_ratio * 100:.2f}%)"

    elif dataset_name == "wikitext103":
        dataset = load_wikitext103()
        display_name = "WikiText-103"

    elif dataset_name == "cc100":
        dataset = load_cc100()
        display_name = "CC100"

    elif dataset_name == "openwebtext":
        dataset = load_openwebtext()
        display_name = "OpenWebText"

    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Choose from: pile, wikitext103, cc100, openwebtext."
        )

    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    return dataset, display_name


def build_dataloader(dataset, batch_size: int, num_workers: int):
    pin_memory = torch.cuda.is_available()

    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": worker_init_fn,
        "generator": get_dataloader_generator(0),
    }

    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = 2
        dataloader_kwargs["persistent_workers"] = False

    return DataLoader(dataset, **dataloader_kwargs)


def infer_checkpoint_dir(checkpoint_path_or_dir: str):
    if os.path.isdir(checkpoint_path_or_dir):
        checkpoint_dir = checkpoint_path_or_dir

        candidates = [
            "best_checkpoint.safetensors",
            "last_checkpoint.safetensors",
        ]

        for name in candidates:
            path = os.path.join(checkpoint_dir, name)
            if os.path.exists(path):
                return checkpoint_dir, path

        raise FileNotFoundError(
            f"No checkpoint file found in {checkpoint_dir}. "
            f"Expected best_checkpoint.safetensors or last_checkpoint.safetensors."
        )

    if os.path.isfile(checkpoint_path_or_dir):
        checkpoint_path = checkpoint_path_or_dir
        checkpoint_dir = os.path.dirname(checkpoint_path)
        return checkpoint_dir, checkpoint_path

    raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path_or_dir}")


def build_gmoe_model_from_checkpoint(
    checkpoint_dir: str,
    model_size: str,
    num_global_experts: int,
    alpha: float,
    capacity_factor: float,
    target_d_ff: int = None,
):
    config_path = os.path.join(checkpoint_dir, "config.json")

    if os.path.exists(config_path):
        config = GPT2Config.from_pretrained(checkpoint_dir)

        num_global_experts = int(
            getattr(config, "num_global_experts", num_global_experts)
        )
        alpha = float(getattr(config, "aux_alpha", alpha))
        capacity_factor = float(
            getattr(config, "capacity_factor", capacity_factor)
        )

        if target_d_ff is None:
            target_d_ff = getattr(config, "n_inner", None)

    else:
        spec = MODEL_SPECS.get(model_size, MODEL_SPECS["base"])

        d_ff = int(target_d_ff) if target_d_ff is not None else int(spec["d_ff"])

        config = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_ctx=1024,
            n_embd=int(spec["n_embd"]),
            n_layer=int(spec["n_layer"]),
            n_head=int(spec["n_head"]),
            n_inner=d_ff,
        )

    if target_d_ff is None:
        target_d_ff = getattr(config, "n_inner", None)

    model = GPT2LMHeadModel(config)

    model = convert_gpt2_to_moe(
        model=model,
        config=config,
        num_global_experts=num_global_experts,
        alpha=alpha,
        capacity_factor=capacity_factor,
        target_d_ff=target_d_ff,
    )

    patch_model_for_gmoe(model)

    return model, config


@torch.no_grad()
def evaluate_ppl(model, dataloader, device, dtype=torch.bfloat16):
    model.eval()

    total_loss_sum = torch.tensor(0.0, device=device)
    total_valid_tokens = torch.tensor(0.0, device=device)

    iterator = tqdm(
        dataloader,
        desc="Evaluating PPL",
        dynamic_ncols=True,
        leave=True,
    )

    use_amp = device.type == "cuda"

    for batch in iterator:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        labels = input_ids.clone()

        pad_mask = (attention_mask == 0) & (labels == enc.eot_token)
        labels[pad_mask] = -100

        if use_amp:
            autocast_ctx = torch.autocast("cuda", dtype=dtype)
        else:
            autocast_ctx = torch.autocast("cpu", enabled=False)

        with autocast_ctx:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
                use_cache=False,
                global_step=(1 << 62),
            )

            logits = outputs.logits

            loss = chunked_cross_entropy(
                logits,
                labels,
                ignore_index=-100,
                chunk_tokens=8192,
            )

        valid_tokens = (labels[:, 1:] != -100).sum()

        total_loss_sum += loss * valid_tokens
        total_valid_tokens += valid_tokens

        running_loss = total_loss_sum / torch.clamp(total_valid_tokens, min=1)
        running_ppl = math.exp(running_loss.item())

        iterator.set_postfix(
            loss=f"{running_loss.item():.4f}",
            ppl=f"{running_ppl:.2f}",
        )

    mean_loss = total_loss_sum / torch.clamp(total_valid_tokens, min=1)
    mean_loss = mean_loss.item()
    ppl = math.exp(mean_loss)

    return mean_loss, ppl


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GMoE checkpoint with PPL.")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to checkpoint file or checkpoint directory. "
            "If omitted, defaults to CHECKPOINTS_DIR/gmoe_base_e16."
        ),
    )

    parser.add_argument(
        "--model_size",
        type=str,
        default="base",
        choices=list(MODEL_SPECS.keys()),
        help="Fallback model size if config.json is missing.",
    )

    parser.add_argument(
        "--num_global_experts",
        type=int,
        default=16,
        help="Fallback number of global experts if config.json is missing.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Evaluation batch size.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of DataLoader workers.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Fallback auxiliary loss coefficient if config.json is missing.",
    )

    parser.add_argument(
        "--capacity_factor",
        type=float,
        default=1.25,
        help="Fallback capacity factor if config.json is missing.",
    )

    parser.add_argument(
        "--target_d_ff",
        type=int,
        default=None,
        help="Fallback FFN dimension if config.json is missing.",
    )

    parser.add_argument(
        "--pile_subset_ratio",
        type=float,
        default=0.01,
        help="Subset ratio for Pile test evaluation.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(42)

    if args.checkpoint is None:
        args.checkpoint = os.path.join(
            CHECKPOINTS_DIR,
            f"gmoe_{args.model_size}_e{args.num_global_experts}",
        )

    checkpoint_dir, checkpoint_path = infer_checkpoint_dir(args.checkpoint)

    print("===== GMoE PPL Evaluation =====")
    print(f"Checkpoint dir : {checkpoint_dir}")
    print(f"Checkpoint     : {checkpoint_path}")
    print(f"Datasets       : {', '.join(EVAL_DATASETS)}")
    print(f"Batch size     : {args.batch_size}")
    print("================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, config = build_gmoe_model_from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        model_size=args.model_size,
        num_global_experts=args.num_global_experts,
        alpha=args.alpha,
        capacity_factor=args.capacity_factor,
        target_d_ff=args.target_d_ff,
    )

    enable_sdp_backends()
    attn_impl = prefer_flash_attention(model)
    print_attn_stack_status(model, tag=f"impl={attn_impl}")

    load_safetensors(
        model=model,
        path=checkpoint_path,
        strict=False,
    )

    model.to(device)
    model.eval()

    model.lm_head.weight = model.transformer.wte.weight

    results = []

    for dataset_name in EVAL_DATASETS:
        print("\n================================")
        print(f"Evaluating dataset: {dataset_name}")
        print("================================")

        dataset, display_name = load_eval_dataset(
            dataset_name=dataset_name,
            pile_subset_ratio=args.pile_subset_ratio,
        )

        dataloader = build_dataloader(
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        loss, ppl = evaluate_ppl(
            model=model,
            dataloader=dataloader,
            device=device,
        )

        results.append(
            {
                "dataset_name": dataset_name,
                "display_name": display_name,
                "loss": loss,
                "ppl": ppl,
            }
        )

        print("\n===== Result =====")
        print(f"Dataset : {display_name}")
        print(f"Loss    : {loss:.4f}")
        print(f"PPL     : {ppl:.2f}")
        print("==================")

    print("\n===== All Results =====")
    for result in results:
        print(
            f"{result['display_name']:20s} | "
            f"Loss: {result['loss']:.4f} | "
            f"PPL: {result['ppl']:.2f}"
        )
    print("=======================")


if __name__ == "__main__":
    main()