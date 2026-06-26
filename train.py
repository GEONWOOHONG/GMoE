import os
import math
import argparse
import contextlib
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import tiktoken

from transformers import GPT2Config, GPT2LMHeadModel
from safetensors.torch import load_model as load_safetensors_model

from modeling import convert_gpt2_to_moe, GPT2LayerMoE
from patches import patch_model_for_gmoe

from config import MODEL_SPECS, CHECKPOINTS_DIR
from data import (
    load_or_prepare_pile,
    worker_init_fn,
    get_dataloader_generator,
)
from utils import (
    set_seed,
    get_default_optimizer,
    get_default_scheduler,
    print_model_info,
    save_checkpoint,
    ensure_flash_attn,
    chunked_cross_entropy,
    enable_sdp_backends,
    prefer_flash_attention,
    print_attn_stack_status,
)


enc = tiktoken.get_encoding("gpt2")


def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        try:
            dist.init_process_group(
                backend="nccl",
                timeout=timedelta(hours=6),
                device_id=torch.device(f"cuda:{local_rank}"),
            )
        except TypeError:
            dist.init_process_group(
                backend="nccl",
                timeout=timedelta(hours=6),
            )

        return True, dist.get_rank(), dist.get_world_size(), local_rank

    return False, 0, 1, 0


def distributed_barrier(local_rank: int):
    if not (dist.is_available() and dist.is_initialized()):
        return

    if torch.cuda.is_available():
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    return (not is_distributed()) or dist.get_rank() == 0


def build_trainer_state(
    optimizer,
    scheduler,
    step,
    best_loss,
    total_steps,
    model_size,
    num_global_experts,
    alpha,
    capacity_factor,
    target_d_ff,
):
    return {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "best_loss": best_loss,
        "total_train_steps": total_steps,
        "model_size": model_size,
        "num_global_experts": num_global_experts,
        "alpha": alpha,
        "capacity_factor": capacity_factor,
        "target_d_ff": target_d_ff,
    }


def build_dataloader(
    dataset,
    batch_size,
    shuffle,
    sampler,
    num_workers,
    generator,
    pin_kwargs,
):
    dataloader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": num_workers,
        "worker_init_fn": worker_init_fn,
        "generator": generator,
        **pin_kwargs,
    }

    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = 8
        dataloader_kwargs["persistent_workers"] = True
    else:
        dataloader_kwargs["prefetch_factor"] = None
        dataloader_kwargs["persistent_workers"] = False

    return DataLoader(**dataloader_kwargs)


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    show_bar=False,
    desc="Valid",
):
    model.eval()

    total_loss_sum = torch.tensor(0.0, device=device)
    total_valid = torch.tensor(0.0, device=device)

    iterator = (
        tqdm(
            dataloader,
            desc=desc,
            leave=False,
            dynamic_ncols=True,
            position=1,
        )
        if show_bar and is_main_process()
        else dataloader
    )

    use_amp = device.type == "cuda"

    for batch in iterator:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        labels = input_ids.clone()
        pad_mask = (attention_mask == 0) & (labels == enc.eot_token)
        labels[pad_mask] = -100

        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if use_amp
            else torch.autocast("cpu", enabled=False)
        )

        with autocast_ctx:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
                use_cache=False,
                global_step=(1 << 62),
            )

            logits = outputs.logits

            batch_mean_loss = chunked_cross_entropy(
                logits,
                labels,
                ignore_index=-100,
                chunk_tokens=8192,
            )

            batch_valid = (labels[:, 1:] != -100).sum()

        total_loss_sum += batch_mean_loss * batch_valid
        total_valid += batch_valid

    if is_distributed():
        dist.all_reduce(total_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_valid, op=dist.ReduceOp.SUM)

    mean_loss = total_loss_sum / torch.clamp(total_valid, min=1)
    mean_loss = mean_loss.item()
    ppl = math.exp(mean_loss)

    model.train()

    return mean_loss, ppl


def compute_gmoe_stats(model):
    moe_layers = [module.moe for module in model.modules() if isinstance(module, GPT2LayerMoE)]

    balance_scores = []

    for moe in moe_layers:
        if getattr(moe, "last_scores", None) is None:
            continue

        scores = moe.last_scores
        probs = scores.mean(dim=0)
        entropy = -(probs * (probs + 1e-9).log()).sum().item()
        balance_scores.append(entropy)

    if not balance_scores:
        return {"balance": 0.0}

    return {
        "balance": sum(balance_scores) / len(balance_scores),
    }


def build_gmoe_model(
    model_size,
    num_global_experts,
    alpha,
    capacity_factor,
    target_d_ff=None,
):
    spec = MODEL_SPECS.get(model_size, MODEL_SPECS["base"])

    if is_main_process():
        print(f"Initializing GPT-2 {model_size.upper()} backbone: {spec}")
        print(f"GMoE global experts: {num_global_experts}")

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

    config.architectures = ["GPT2LMHeadModel"]
    config.model_type = "gpt2"
    config.moe_type = "gmoe"
    config.num_global_experts = int(num_global_experts)
    config.capacity_factor = float(capacity_factor)
    config.aux_alpha = float(alpha)
    config.loss_type = "ForCausalLMLoss"

    model = GPT2LMHeadModel(config)

    model = convert_gpt2_to_moe(
        model=model,
        config=config,
        num_global_experts=num_global_experts,
        alpha=alpha,
        capacity_factor=capacity_factor,
        target_d_ff=d_ff,
    )

    patch_model_for_gmoe(model)

    return model, config


def load_training_state(
    model,
    optimizer,
    scheduler,
    checkpoint_dir,
    checkpoint_name="best_checkpoint.safetensors",
):
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    trainer_path = os.path.join(checkpoint_dir, "best_checkpoint_trainer.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not os.path.exists(trainer_path):
        raise FileNotFoundError(f"Trainer state not found: {trainer_path}")

    load_safetensors_model(model, checkpoint_path, strict=False)
    trainer_state = torch.load(trainer_path, map_location="cpu")

    optimizer.load_state_dict(trainer_state["optimizer"])

    if "scheduler" in trainer_state:
        scheduler.load_state_dict(trainer_state["scheduler"])

    start_step = int(trainer_state.get("step", 0))
    best_loss = float(trainer_state.get("best_loss", float("inf")))

    if is_main_process():
        print(f"Loaded checkpoint: {checkpoint_path}")
        print(f"Resumed from step={start_step}, best_loss={best_loss:.4f}")

    return start_step, best_loss


def train_gmoe(
    model_size="base",
    num_global_experts=16,
    batch_size=32,
    seq_len=1024,
    grad_accum=1,
    alpha=0.01,
    capacity_factor=1.25,
    target_d_ff=None,
    output_dir=None,
    resume=False,
    num_workers=8,
):
    is_dist, rank, world_size, local_rank = init_distributed()

    set_seed(42)

    if is_main_process():
        ensure_flash_attn()

    if is_dist:
        distributed_barrier(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if output_dir is None:
        output_dir = os.path.join(
            CHECKPOINTS_DIR,
            f"gmoe_{model_size}_e{num_global_experts}",
        )

    os.makedirs(output_dir, exist_ok=True)

    if is_main_process():
        print(f"Checkpoint directory: {output_dir}")

    model, config = build_gmoe_model(
        model_size=model_size,
        num_global_experts=num_global_experts,
        alpha=alpha,
        capacity_factor=capacity_factor,
        target_d_ff=target_d_ff,
    )

    enable_sdp_backends()
    attn_impl = prefer_flash_attention(model)

    if is_main_process():
        print_attn_stack_status(model, tag=f"impl={attn_impl}")
        config.save_pretrained(output_dir)

    train_dataset, valid_dataset = load_or_prepare_pile(verbose=is_main_process())
    valid_dataset = valid_dataset.select(range(int(0.1 * len(valid_dataset))))

    if is_main_process():
        print(
            f"Using datasets: train={len(train_dataset):,}, "
            f"valid={len(valid_dataset):,}"
        )

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    valid_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(train_dataset)
        if is_dist
        else None
    )

    valid_sampler = (
        torch.utils.data.distributed.DistributedSampler(
            valid_dataset,
            shuffle=False,
        )
        if is_dist
        else None
    )

    train_generator = get_dataloader_generator(rank if is_dist else 0)
    valid_generator = get_dataloader_generator(rank if is_dist else 0)

    pin_kwargs = {}

    if torch.cuda.is_available():
        pin_kwargs["pin_memory"] = True

        try:
            from packaging import version

            if version.parse(torch.__version__) >= version.parse("1.12.1"):
                pin_kwargs["pin_memory_device"] = f"cuda:{local_rank}"
        except Exception:
            pass

    if is_main_process():
        print(
            f"DataLoader workers per GPU: {num_workers}, "
            f"pin_memory={pin_kwargs.get('pin_memory', False)}"
        )

    train_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        generator=train_generator,
        pin_kwargs=pin_kwargs,
    )

    valid_loader = build_dataloader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=valid_sampler,
        num_workers=num_workers,
        generator=valid_generator,
        pin_kwargs=pin_kwargs,
    )

    num_batches = len(train_loader)
    total_steps = math.ceil(num_batches / grad_accum)

    if is_main_process():
        print_model_info(
            model,
            config,
            "gmoe",
            num_global_experts,
            batch_size=batch_size,
            grad_accum_steps=grad_accum,
            effective_batch=batch_size * world_size * grad_accum,
        )

        tokens_per_optim_step = batch_size * world_size * grad_accum * seq_len
        total_tokens = total_steps * tokens_per_optim_step

        print(f"FFN dimension: {config.n_inner}")
        print(f"Training steps: {total_steps:,}")
        print(f"Tokens per optimizer step: {tokens_per_optim_step:,}")
        print(f"Approx. total tokens: {total_tokens / 1e9:.2f}B")

    model.to(device)

    base_model = model

    if is_dist:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
            find_unused_parameters=False,
            static_graph=False,
            bucket_cap_mb=100,
        )

    optimizer = get_default_optimizer(model)

    scheduler = get_default_scheduler(
        optimizer,
        total_steps,
        warmup_ratio=0.1,
    )

    if is_dist:
        try:
            from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as dh

            model.register_comm_hook(state=None, hook=dh.fp16_compress_hook)
        except Exception:
            pass

    writer = None

    if is_main_process():
        runs_root = os.path.join(output_dir, "runs")
        os.makedirs(runs_root, exist_ok=True)
        writer = SummaryWriter(log_dir=runs_root)

    start_step = 0
    best_loss = float("inf")

    if resume:
        start_step, best_loss = load_training_state(
            model=base_model,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_dir=output_dir,
            checkpoint_name="best_checkpoint.safetensors",
        )

    progress_bar = (
        tqdm(
            total=total_steps,
            initial=start_step,
            desc=f"Training rank {rank}",
            dynamic_ncols=True,
            leave=True,
        )
        if is_main_process()
        else None
    )

    full_eval_interval = max(1, total_steps // 10)

    optim_step = start_step
    reached_budget = False
    use_amp = device.type == "cuda"

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1):
        if is_dist and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for micro_step, batch in enumerate(train_loader, start=1):
            if optim_step >= total_steps:
                reached_budget = True
                break

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            labels = input_ids.clone()
            pad_mask = (attention_mask == 0) & (labels == enc.eot_token)
            labels[pad_mask] = -100

            sync_now = (micro_step % grad_accum) == 0

            sync_context = (
                model.no_sync()
                if is_dist and not sync_now
                else contextlib.nullcontext()
            )

            autocast_ctx = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if use_amp
                else torch.autocast("cpu", enabled=False)
            )

            with sync_context:
                with autocast_ctx:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=None,
                        use_cache=False,
                        global_step=optim_step,
                    )

                    logits = outputs.logits

                    main_loss = chunked_cross_entropy(
                        logits,
                        labels,
                        ignore_index=-100,
                        chunk_tokens=8192,
                    )

                balance_losses = [
                    module.last_balance_loss
                    for module in base_model.modules()
                    if isinstance(module, GPT2LayerMoE)
                    and module.last_balance_loss is not None
                ]

                if balance_losses:
                    aux_loss = torch.stack(balance_losses).mean()
                else:
                    aux_loss = torch.zeros(
                        (),
                        device=main_loss.device,
                        dtype=main_loss.dtype,
                    )

                loss = main_loss + aux_loss
                (loss / grad_accum).backward()

            if not sync_now:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            optim_step += 1

            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix(
                    loss=float(loss.detach().item()),
                    main=float(main_loss.detach().item()),
                    aux=float(aux_loss.detach().item()),
                )

            if writer is not None and optim_step % 50 == 0:
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], optim_step)
                writer.add_scalar("train/loss_total", float(loss.detach().item()), optim_step)
                writer.add_scalar("train/loss_main", float(main_loss.detach().item()), optim_step)
                writer.add_scalar("train/loss_aux", float(aux_loss.detach().item()), optim_step)

            should_full_eval = optim_step % full_eval_interval == 0

            if should_full_eval:
                if progress_bar is not None:
                    progress_bar.write(f"Full validation at step {optim_step}")

                valid_loss, valid_ppl = evaluate(
                    model=model,
                    dataloader=valid_loader,
                    device=device,
                    show_bar=True,
                    desc=f"Valid @ step {optim_step}",
                )

                stats = compute_gmoe_stats(base_model)

                if is_main_process():
                    print(
                        f"[Step {optim_step}] "
                        f"Valid loss={valid_loss:.4f}, "
                        f"PPL={valid_ppl:.2f}, "
                        f"Balance={stats['balance']:.4f}"
                    )

                    if writer is not None:
                        writer.add_scalar("valid/loss", valid_loss, optim_step)
                        writer.add_scalar("valid/ppl", valid_ppl, optim_step)
                        writer.add_scalar("valid/balance", stats["balance"], optim_step)

                    if valid_loss < best_loss:
                        best_loss = valid_loss

                        save_checkpoint(
                            base_model,
                            optimizer,
                            scheduler,
                            optim_step,
                            best_loss,
                            total_steps,
                            output_dir,
                            "best_checkpoint.safetensors",
                        )

                        trainer_state = build_trainer_state(
                            optimizer=optimizer,
                            scheduler=scheduler,
                            step=optim_step,
                            best_loss=best_loss,
                            total_steps=total_steps,
                            model_size=model_size,
                            num_global_experts=num_global_experts,
                            alpha=alpha,
                            capacity_factor=capacity_factor,
                            target_d_ff=target_d_ff,
                        )

                        torch.save(
                            trainer_state,
                            os.path.join(output_dir, "best_checkpoint_trainer.pt"),
                        )

            if optim_step >= total_steps:
                reached_budget = True
                break

        if reached_budget:
            break

    if progress_bar is not None:
        progress_bar.close()

    if is_main_process():
        save_checkpoint(
            base_model,
            optimizer,
            scheduler,
            optim_step,
            best_loss,
            total_steps,
            output_dir,
            "last_checkpoint.safetensors",
        )

        final_state = build_trainer_state(
            optimizer=optimizer,
            scheduler=scheduler,
            step=optim_step,
            best_loss=best_loss,
            total_steps=total_steps,
            model_size=model_size,
            num_global_experts=num_global_experts,
            alpha=alpha,
            capacity_factor=capacity_factor,
            target_d_ff=target_d_ff,
        )

        torch.save(
            final_state,
            os.path.join(output_dir, "last_checkpoint_trainer.pt"),
        )

        if writer is not None:
            writer.close()

        print(f"Training finished at step {optim_step}/{total_steps}")
        print(f"Best valid loss: {best_loss:.4f}")
        print(f"Saved to: {output_dir}")

    if is_dist:
        distributed_barrier(local_rank)
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="Train GMoE.")

    parser.add_argument(
        "--model_size",
        type=str,
        default="base",
        choices=list(MODEL_SPECS.keys()),
        help="Model size defined in config.MODEL_SPECS.",
    )

    parser.add_argument(
        "--num_global_experts",
        type=int,
        default=16,
        help="Number of global experts shared across all layers.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Per-GPU batch size.",
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        default=1024,
        help="Sequence length.",
    )

    parser.add_argument(
        "--grad_accum",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Auxiliary load-balancing loss coefficient.",
    )

    parser.add_argument(
        "--capacity_factor",
        type=float,
        default=1.25,
        help="Checkpoint-compatible capacity factor field.",
    )

    parser.add_argument(
        "--target_d_ff",
        type=int,
        default=None,
        help="Optional FFN hidden dimension.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Checkpoint output directory.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from best checkpoint in output_dir.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of DataLoader workers per GPU.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_gmoe(
        model_size=args.model_size,
        num_global_experts=args.num_global_experts,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        grad_accum=args.grad_accum,
        alpha=args.alpha,
        capacity_factor=args.capacity_factor,
        target_d_ff=args.target_d_ff,
        output_dir=args.output_dir,
        resume=args.resume,
        num_workers=args.num_workers,
    )