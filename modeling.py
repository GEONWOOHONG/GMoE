# modeling.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config
from typing import Optional


class Expert(nn.Module):
    def __init__(self, d_model, d_ff, initializer_range=0.02, use_gelu=False):
        super().__init__()

        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.GELU() if use_gelu else nn.ReLU()

        self._reset_parameters(initializer_range)

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))

    def _reset_parameters(self, initializer_range):
        nn.init.normal_(self.w1.weight, mean=0.0, std=initializer_range)
        nn.init.normal_(self.w2.weight, mean=0.0, std=initializer_range)


class RecurrentRouter(nn.Module):
    def __init__(self, d_model, hidden_dim=None):
        super().__init__()

        self.hidden_dim = hidden_dim or d_model

        if self.hidden_dim != d_model:
            raise ValueError(
                "This public checkpoint-compatible RecurrentRouter assumes "
                "hidden_dim == d_model because no input projection is used."
            )

        self.gru = nn.GRUCell(d_model, self.hidden_dim)

        nn.init.xavier_uniform_(self.gru.weight_ih)
        nn.init.orthogonal_(self.gru.weight_hh)
        nn.init.zeros_(self.gru.bias_ih)
        nn.init.zeros_(self.gru.bias_hh)

    def forward(self, x, h_prev=None):
        if x.dim() == 3:
            bsz, seq_len, d_model = x.shape
            x = x.reshape(bsz * seq_len, d_model)

        if h_prev is None:
            h_prev = torch.zeros(
                x.size(0),
                self.hidden_dim,
                device=x.device,
                dtype=x.dtype,
            )

        return self.gru(x, h_prev)


class GMoELayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
        num_global_experts,
        alpha=0.01,
        capacity_factor=1.25,
        global_experts=None,
        shared_router: RecurrentRouter = None,
    ):
        super().__init__()

        if global_experts is None:
            raise ValueError("global_experts must not be None")
        if shared_router is None:
            raise ValueError("shared_router must not be None")
        if num_global_experts < 1:
            raise ValueError("GMoE requires at least one global expert")

        self.mode = "gmoe"

        self.d_model = int(d_model)
        self.aux_alpha = float(alpha)
        self.capacity_factor = float(capacity_factor)

        self.experts = nn.ModuleList([Expert(d_model, d_ff)])

        self.global_experts = global_experts
        self.num_experts = len(global_experts)

        if self.num_experts != int(num_global_experts):
            raise ValueError("num_global_experts must match len(global_experts)")

        self.shared_router = shared_router
        self.cond_dim = self.shared_router.gru.hidden_size

        self.h_ln = nn.LayerNorm(self.cond_dim)

        self.prev_score_ln = nn.LayerNorm(self.num_experts)
        self.prev_score_proj = nn.Linear(self.num_experts, self.cond_dim, bias=False)

        self.gate_head = nn.Linear(
            self.cond_dim + self.cond_dim,
            self.num_experts,
            bias=False,
        )

        self.last_scores = None

    def _compute_aux_loss(self, scores, top_idx_flat):
        one_hot = F.one_hot(top_idx_flat, num_classes=self.num_experts).float()
        freq = one_hot.mean(dim=0)

        pi = scores.mean(dim=0)
        fi = freq * self.num_experts

        return (pi * fi).sum() * self.aux_alpha

    def _ddp_probe_loss(self, dtype, device, alpha=1e-8):
        d = self.d_model
        probe = torch.randn(8, d, device=device, dtype=dtype) * 1e-3

        loss = (self.experts[0](probe) ** 2).mean()

        for expert in self.global_experts:
            loss = loss + (expert(probe) ** 2).mean()

        return alpha * loss

    def forward(self, x, routing_state=None, global_step: Optional[int] = None):
        del global_step

        bsz, seq_len, hidden_dim = x.shape
        x_flat = x.reshape(-1, hidden_dim)
        num_tokens = x_flat.size(0)

        h_prev = None
        prev_logits = None

        if isinstance(routing_state, dict):
            h_prev = routing_state.get("h", None)
            prev_logits = routing_state.get("logits", None)

        h_new = self.shared_router(x, h_prev=h_prev)

        h_norm = self.h_ln(h_new)
        h_feat = h_norm.view(-1, h_norm.size(-1))

        if prev_logits is None:
            prev_logits = h_feat.new_zeros((num_tokens, self.num_experts))
        else:
            prev_logits = prev_logits.detach().to(h_feat.dtype)

        prev_feat = self.prev_score_proj(self.prev_score_ln(prev_logits))

        gate_in = torch.cat([h_feat, prev_feat], dim=-1)
        logits = self.gate_head(gate_in)

        scores = F.softmax(logits, dim=-1)
        self.last_scores = scores.detach()

        local_out = self.experts[0](x)

        top_scores, top_idx = torch.topk(scores, k=1, dim=-1)

        top_idx_flat = top_idx.view(-1)
        top_scores_flat = top_scores.view(-1, 1)

        global_out_flat = torch.zeros_like(x_flat)

        for expert_id, expert in enumerate(self.global_experts):
            token_idx = (top_idx_flat == expert_id).nonzero(as_tuple=True)[0]

            if token_idx.numel() == 0:
                continue

            expert_out = expert(x_flat[token_idx]) * top_scores_flat[token_idx]
            global_out_flat[token_idx] = expert_out.to(global_out_flat.dtype)

        global_out = global_out_flat.view_as(x)

        routed_out = local_out + global_out

        aux_loss = None

        if self.training and self.aux_alpha > 0.0:
            aux_loss = self._compute_aux_loss(scores, top_idx_flat)

        probe_alpha = getattr(self, "ddp_probe_alpha", 1e-8)
        probe_loss = self._ddp_probe_loss(
            dtype=x.dtype,
            device=x.device,
            alpha=probe_alpha,
        )

        aux_loss = probe_loss if aux_loss is None else aux_loss + probe_loss

        updated_routing_state = {
            "h": h_new,
            "logits": logits,
        }

        return routed_out, aux_loss, updated_routing_state


class GPT2LayerMoE(nn.Module):
    def __init__(
        self,
        config: GPT2Config,
        num_global_experts,
        global_experts=None,
        alpha=0.01,
        capacity_factor=1.25,
        layer_idx=None,
        shared_router: RecurrentRouter = None,
        d_ff=None,
    ):
        super().__init__()

        if d_ff is None:
            cfg_inner = getattr(config, "n_inner", None)
            d_ff = int(cfg_inner) if cfg_inner is not None else int(config.n_embd * 4)

        self.mode = "gmoe"

        self.moe = GMoELayer(
            d_model=config.n_embd,
            d_ff=d_ff,
            num_global_experts=num_global_experts,
            global_experts=global_experts,
            alpha=alpha,
            capacity_factor=capacity_factor,
            shared_router=shared_router,
        )

        self.layer_idx = 0 if layer_idx is None else int(layer_idx)
        setattr(self.moe, "_layer_idx", self.layer_idx)

        self.last_balance_loss = None

    def forward(self, hidden_states, routing_state=None, global_step=None, **kwargs):
        del kwargs

        out, balance_loss, updated_routing_state = self.moe(
            hidden_states,
            routing_state=routing_state,
            global_step=global_step,
        )

        self.last_balance_loss = balance_loss

        return out, balance_loss, updated_routing_state


def convert_gpt2_to_moe(
    model,
    config,
    num_global_experts,
    alpha=0.01,
    capacity_factor=1.25,
    target_d_ff=None,
):
    if num_global_experts < 1:
        raise ValueError("GMoE requires at least one global expert")

    if target_d_ff is not None:
        eff_d_ff = int(target_d_ff)
    else:
        cfg_inner = getattr(config, "n_inner", None)
        eff_d_ff = int(cfg_inner) if cfg_inner is not None else int(config.n_embd * 4)

    global_experts = nn.ModuleList(
        [
            Expert(
                config.n_embd,
                eff_d_ff,
                initializer_range=config.initializer_range,
            )
            for _ in range(num_global_experts)
        ]
    )

    shared_router = RecurrentRouter(
        d_model=config.n_embd,
        hidden_dim=config.n_embd,
    )

    for layer_idx, block in enumerate(model.transformer.h):
        block.mlp = GPT2LayerMoE(
            config=config,
            num_global_experts=num_global_experts,
            d_ff=eff_d_ff,
            global_experts=global_experts,
            alpha=alpha,
            capacity_factor=capacity_factor,
            shared_router=shared_router,
            layer_idx=layer_idx,
        )

    return model