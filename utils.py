# utils.py
import os
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from safetensors.torch import save_file, safe_open
from transformers import get_cosine_schedule_with_warmup


def _is_rank0() -> bool:
    return os.environ.get("RANK", "0") == "0"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.set_float32_matmul_precision("high")


def get_default_optimizer(model):
    params = (p for p in model.parameters() if p.requires_grad)

    try:
        optimizer = optim.AdamW(
            params,
            lr=3e-4,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            fused=True,
        )

        if _is_rank0():
            print("Optimizer: AdamW (fused=True)")

        return optimizer

    except TypeError:
        optimizer = optim.AdamW(
            params,
            lr=3e-4,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )

        if _is_rank0():
            print("Optimizer: AdamW (fused unsupported -> fallback)")

        return optimizer


def get_default_scheduler(optimizer, total_steps, warmup_ratio=0.1):
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    if _is_rank0():
        print(
            f"Scheduler: cosine "
            f"(warmup_ratio={warmup_ratio:.3f}, "
            f"warmup_steps={warmup_steps}, "
            f"total_steps={total_steps})"
        )

    return scheduler


def enable_sdp_backends():
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)

    os.environ.setdefault("PYTORCH_CUDA_SDPA_ALLOW_FLASH_ATTENTION", "1")


def prefer_flash_attention(model):
    try:
        model.config.attn_implementation = "flash_attention_2"
        return "flash_attention_2"
    except Exception:
        pass

    try:
        model.config._attn_implementation = "flash_attention_2"
        return "flash_attention_2"
    except Exception:
        pass

    try:
        model.config.attn_implementation = "sdpa"
        return "sdpa"
    except Exception:
        pass

    try:
        model.config._attn_implementation = "sdpa"
        return "sdpa"
    except Exception:
        return "unknown"


def print_attn_stack_status(model, tag=""):
    if not torch.cuda.is_available():
        print(f"[ATTN {tag}] CUDA is not available.")
        return

    device_props = torch.cuda.get_device_properties(0)

    try:
        from transformers.utils import is_flash_attn_2_available

        flash_attn_2_available = is_flash_attn_2_available()
    except Exception:
        flash_attn_2_available = "unknown"

    impl = getattr(
        getattr(model, "config", object()),
        "attn_implementation",
        getattr(
            getattr(model, "config", object()),
            "_attn_implementation",
            "unset",
        ),
    )

    print(
        f"[ATTN {tag}] "
        f"device={device_props.name} "
        f"sm={device_props.major}{device_props.minor} "
        f"torch={torch.__version__} "
        f"cuda={torch.version.cuda}"
    )

    print(
        f"[ATTN {tag}] "
        f"flash_sdp={torch.backends.cuda.flash_sdp_enabled()} "
        f"mem_efficient_sdp={torch.backends.cuda.mem_efficient_sdp_enabled()} "
        f"math_sdp={torch.backends.cuda.math_sdp_enabled()} "
        f"flash_attn_2_available={flash_attn_2_available}"
    )

    print(f"[ATTN {tag}] model.config.attn_implementation={impl}")


def ensure_flash_attn():
    import importlib.util

    spec = importlib.util.find_spec("flash_attn")

    if spec is not None:
        print("FlashAttention is installed.")
    else:
        print("FlashAttention is not installed. Falling back to SDPA if available.")


def chunked_cross_entropy(
    logits,
    labels,
    ignore_index=-100,
    chunk_tokens=8192,
):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    batch_size, seq_len_minus_1, vocab_size = shift_logits.shape
    total_tokens = batch_size * seq_len_minus_1

    flat_logits = shift_logits.view(total_tokens, vocab_size)
    flat_labels = shift_labels.view(total_tokens)

    total_loss = torch.zeros(
        (),
        device=logits.device,
        dtype=torch.float32,
    )

    valid_tokens = 0

    for start in range(0, total_tokens, chunk_tokens):
        end = min(start + chunk_tokens, total_tokens)

        loss = F.cross_entropy(
            flat_logits[start:end],
            flat_labels[start:end],
            ignore_index=ignore_index,
            reduction="sum",
        )

        total_loss = total_loss + loss.float()

        if ignore_index is not None:
            valid_tokens += (flat_labels[start:end] != ignore_index).sum().item()
        else:
            valid_tokens += end - start

    denom = max(valid_tokens, 1)
    return total_loss / denom


@torch.no_grad()
def save_checkpoint(
    model,
    optimizer,
    scheduler,
    step,
    best_loss,
    total_steps,
    save_dir,
    filename,
):
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, filename)
    state_dict = model.state_dict()

    cloned_state_dict = {}

    for key, value in state_dict.items():
        if (
            "global_experts" in key
            or "shared_router" in key
            or "lm_head.weight" in key
            or "transformer.wte.weight" in key
        ):
            cloned_state_dict[key] = value.clone().detach()
        else:
            cloned_state_dict[key] = value.detach()

    save_file(cloned_state_dict, model_path)

    if _is_rank0():
        print(f"[save_checkpoint] Saved model weights to: {model_path}")


@torch.no_grad()
def load_safetensors(model, path: str, strict: bool = False):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    model_state = model.state_dict()
    model_keys = set(model_state.keys())

    loaded_keys = set()

    with safe_open(path, framework="pt", device="cpu") as f:
        checkpoint_keys = set(f.keys())

        for key in checkpoint_keys:
            if key not in model_keys:
                continue

            tensor = f.get_tensor(key)

            if model_state[key].shape != tensor.shape:
                if strict:
                    raise RuntimeError(
                        f"Shape mismatch for key={key}: "
                        f"model={tuple(model_state[key].shape)}, "
                        f"checkpoint={tuple(tensor.shape)}"
                    )
                continue

            model_state[key].copy_(tensor)
            loaded_keys.add(key)

    missing_keys = sorted(model_keys - loaded_keys)
    unexpected_keys = sorted(checkpoint_keys - model_keys)

    if _is_rank0():
        print(f"Loaded {len(loaded_keys)}/{len(model_keys)} tensors from {path}")

        if missing_keys:
            print(f"Missing keys: {len(missing_keys)}")
            for key in missing_keys[:10]:
                print(f"  - {key}")
            if len(missing_keys) > 10:
                print("  ...")

        if unexpected_keys:
            print(f"Unexpected keys: {len(unexpected_keys)}")
            for key in unexpected_keys[:10]:
                print(f"  - {key}")
            if len(unexpected_keys) > 10:
                print("  ...")

    if strict and missing_keys:
        raise RuntimeError(f"Missing keys when loading checkpoint: {missing_keys[:20]}")


def compute_gmoe_active_params(model, config):
    n_layer = int(config.n_layer)
    d_model = int(config.n_embd)
    vocab_size = int(config.vocab_size)
    max_pos = int(config.n_positions)

    d_ff = getattr(config, "n_inner", None)
    if d_ff is None:
        d_ff = 4 * d_model
    d_ff = int(d_ff)

    embedding_params = vocab_size * d_model + max_pos * d_model

    active_layer_params = 0

    for _ in range(n_layer):
        attention_params = 4 * (d_model ** 2)
        active_expert_params = 2 * (2 * d_model * d_ff)

        active_layer_params += attention_params + active_expert_params

    return embedding_params + active_layer_params


def print_model_info(
    model,
    config,
    mode="gmoe",
    num_experts=None,
    batch_size=None,
    grad_accum_steps=None,
    effective_batch=None,
):
    print("===== Model Configuration =====")
    print(f"Backbone        : GPT-2 ({config.n_layer} layers)")
    print(f"Model Type      : {mode}")
    print(f"Hidden Dim      : {config.n_embd}")
    print(f"FFN Dim         : {getattr(config, 'n_inner', config.n_embd * 4)}")
    print(f"Attention Heads : {config.n_head}")
    print(f"Vocab Size      : {config.vocab_size}")

    if num_experts is not None:
        print(f"Global Experts  : {num_experts}")

    print("--------------------------------")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    active_params = compute_gmoe_active_params(model, config)

    print(f"Total Params    : {total_params / 1e6:.2f}M")
    print(f"Trainable       : {trainable_params / 1e6:.2f}M")
    print(f"Active Params   : {active_params / 1e6:.2f}M")

    if total_params > 0:
        print(f"Active Ratio    : {active_params / total_params:.2%}")

    print("--------------------------------")

    from modeling import GPT2LayerMoE

    moe_layers = [module for module in model.modules() if isinstance(module, GPT2LayerMoE)]

    print(f"GMoE Layers     : {len(moe_layers)}")

    for idx, layer in enumerate(moe_layers[:4]):
        moe = layer.moe
        print(
            f"  - Layer {idx}: "
            f"local=1, "
            f"global={len(moe.global_experts)}, "
            f"top_k=1, "
            f"shared_router=True"
        )

    if len(moe_layers) > 4:
        print(f"  ... {len(moe_layers) - 4} more layers")

    if (
        batch_size is not None
        and grad_accum_steps is not None
        and effective_batch is not None
    ):
        print(
            f"Batch per GPU   : {batch_size}, "
            f"Grad Accum      : {grad_accum_steps}, "
            f"Effective Batch : {effective_batch}"
        )

    print("================================")