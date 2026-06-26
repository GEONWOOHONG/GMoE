#patches.py
import os
import types
import torch
from transformers import GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from modeling import GPT2LayerMoE


def create_causal_mask(config, input_embeds, attention_mask=None, cache_position=None, past_key_values=None, position_ids=None):
    del config, cache_position, past_key_values, position_ids

    bsz, seq_len, _ = input_embeds.size()
    device = input_embeds.device
    dtype = input_embeds.dtype

    causal = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))
    causal = causal.view(1, 1, seq_len, seq_len)

    if attention_mask is None:
        allowed = causal.expand(bsz, 1, seq_len, seq_len)
    else:
        k_mask = attention_mask.bool().view(bsz, 1, 1, seq_len)
        q_mask = attention_mask.bool().view(bsz, 1, seq_len, 1)
        allowed = causal & k_mask & q_mask
        self_edge = torch.eye(seq_len, device=device, dtype=torch.bool).view(1, 1, seq_len, seq_len)
        allowed = torch.where(q_mask, allowed, self_edge)

    neg_inf = torch.finfo(torch.float32).min
    additive = (~allowed).to(torch.float32) * neg_inf
    return additive.to(dtype)


def block_moe_forward_patch(
    self,
    hidden_states: torch.FloatTensor,
    past_key_values=None,
    cache_position=None,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    use_cache=False,
    output_attentions=False,
    routing_state=None,
    **kwargs,
):
    kwargs.pop("position_ids", None)
    kwargs.pop("layer_past", None)
    kwargs.pop("use_cache", None)
    kwargs.pop("output_attentions", None)

    global_step = kwargs.get("global_step", None)
    if global_step is None:
        global_step = getattr(self, "_temp_global_step", None)

    residual = hidden_states
    hidden_states = self.ln_1(hidden_states)

    attn_outputs = self.attn(
        hidden_states,
        past_key_values=past_key_values,
        cache_position=cache_position,
        attention_mask=attention_mask,
        head_mask=head_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
    )

    attn_output = attn_outputs[0]

    present = None
    attn_probs = None

    if len(attn_outputs) == 2:
        second = attn_outputs[1]
        if isinstance(second, (tuple, list)) or (hasattr(second, "shape") and getattr(second, "ndim", None) in (3,)):
            present = second
        else:
            attn_probs = second
    elif len(attn_outputs) >= 3:
        present = attn_outputs[1]
        attn_probs = attn_outputs[2]

    hidden_states = residual + attn_output

    residual = hidden_states
    normed = self.ln_2(hidden_states)

    if isinstance(self.mlp, GPT2LayerMoE):
        mlp_result = self.mlp(
            normed,
            routing_state=routing_state,
            global_step=global_step,
        )
    else:
        mlp_result = self.mlp(normed)

    if isinstance(mlp_result, tuple):
        out, balance_loss, updated_routing_state = mlp_result
    else:
        out = mlp_result
        balance_loss = None
        updated_routing_state = routing_state

    try:
        self.mlp.last_balance_loss = balance_loss
    except Exception:
        pass

    hidden_states = residual + out

    ret = (hidden_states,)
    if present is not None:
        ret += (present,)
    if output_attentions and attn_probs is not None:
        ret += (attn_probs,)
    ret += (updated_routing_state,)
    return ret


def patch_model_basic(model: GPT2LMHeadModel):
    GPT2Block.forward = block_moe_forward_patch

    original_lm_forward = model.forward

    def patched_lm_forward(self, input_ids=None, **kwargs):
        all_args = {"input_ids": input_ids, **kwargs}
        global_step = kwargs.get("global_step", None)

        if input_ids is not None:
            for block in self.transformer.h:
                if global_step is not None:
                    setattr(block, "_temp_global_step", global_step)

        outputs = original_lm_forward(**all_args)

        for block in self.transformer.h:
            if hasattr(block, "_temp_global_step"):
                delattr(block, "_temp_global_step")

        return outputs

    model.forward = types.MethodType(patched_lm_forward, model)


def patch_model_for_gmoe(model: GPT2LMHeadModel):
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        print("Applying GMoE forward patches...")
    patch_model_basic(model)

    def patched_model_forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        seq_length = input_shape[-1]
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if attention_mask is not None and attention_mask.ndim < 4:
            attention_mask = attention_mask.view(batch_size, -1)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        if token_type_ids is not None:
            hidden_states = hidden_states + self.wte(token_type_ids)

        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)

        routing_state = None
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            global_step = kwargs.get("global_step", None)

            outputs = block(
                hidden_states,
                past_key_values=past_key_values[i] if past_key_values is not None else None,
                cache_position=cache_position,
                attention_mask=causal_mask,
                head_mask=head_mask[i] if head_mask is not None else None,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                routing_state=routing_state,
                global_step=global_step,
            )

            hidden_states = outputs[0]
            routing_state = outputs[-1]

            if output_attentions:
                attn = outputs[-2]
                all_self_attentions = all_self_attentions + (attn,)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, None, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=None,
        )

    model.transformer.forward = types.MethodType(patched_model_forward, model.transformer)
    return model