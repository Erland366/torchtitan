"""Inference-only nanoVLM generation runtime with KV-cache for eval.

This module intentionally lives under ``torchtitan.eval`` so downstream eval can
stay standalone without importing ``nanoVLM_main`` at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from torchtitan.models.nanoVLM.attention import (
    create_base_structural_block_mask,
    create_causal_block_mask,
    create_momh_block_mask_from_modality,
    flex_attention_compiled,
    flex_attention_compiled_dynamic,
    generate_soft_gating_score_mod,
)
from torchtitan.models.nanoVLM.model import NanoVLMModel, apply_rotary_pos_embd


@dataclass
class _LayerCache:
    key: torch.Tensor
    value: torch.Tensor


class NanoVLMCachedGenerator:
    """KV-cache generation helper for ``NanoVLMModel`` inference."""

    def __init__(self, model: NanoVLMModel, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _build_shared_block_mask(
        self,
        *,
        attn,
        q_len: int,
        kv_len: int,
        full_is_vision: torch.Tensor,
        full_attention_mask: torch.Tensor,
        device: torch.device,
    ):
        if not (
            attn.momh_enabled
            and full_attention_mask is not None
            and device.type == "cuda"
            and kv_len > 1
        ):
            return None

        if attn.momh_soft_gating or attn.momh_structural_mask_only:
            if attn.momh_causal_gating:
                return create_causal_block_mask(
                    q_len=q_len,
                    kv_len=kv_len,
                    attention_mask=full_attention_mask[:, :kv_len],
                    device=str(device),
                )
            return create_base_structural_block_mask(
                q_len=q_len,
                kv_len=kv_len,
                is_vision=full_is_vision[:, :kv_len],
                attention_mask=full_attention_mask[:, :kv_len],
                device=str(device),
            )

        return create_momh_block_mask_from_modality(
            n_q_heads=int(attn.n_heads),
            q_len=q_len,
            kv_len=kv_len,
            is_vision=full_is_vision[:, :kv_len],
            attention_mask=full_attention_mask[:, :kv_len],
            n_v_heads=int(attn.momh_n_v_heads),
            n_t_heads=int(attn.momh_n_t_heads),
            device=str(device),
        )

    def _attention_with_cache(
        self,
        *,
        attn,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        full_attention_mask: torch.Tensor,
        full_is_vision: torch.Tensor,
        layer_cache: _LayerCache | None,
        prefill_block_mask,
    ) -> tuple[torch.Tensor, _LayerCache]:
        batch_size, t_curr, dim = x.size()

        q = (
            attn.q_proj(x)
            .view(batch_size, t_curr, attn.n_heads, attn.head_dim)
            .transpose(1, 2)
        )
        k_curr = (
            attn.k_proj(x)
            .view(batch_size, t_curr, attn.n_kv_heads, attn.head_dim)
            .transpose(1, 2)
        )
        v_curr = (
            attn.v_proj(x)
            .view(batch_size, t_curr, attn.n_kv_heads, attn.head_dim)
            .transpose(1, 2)
        )

        q, k_rot = apply_rotary_pos_embd(q, k_curr, cos, sin)

        if layer_cache is None:
            k_cat = k_rot
            v_cat = v_curr
        else:
            k_cat = torch.cat((layer_cache.key, k_rot), dim=2)
            v_cat = torch.cat((layer_cache.value, v_curr), dim=2)
        updated_cache = _LayerCache(key=k_cat, value=v_cat)

        k_exp = k_cat.repeat_interleave(attn.n_kv_groups, dim=1)
        v_exp = v_cat.repeat_interleave(attn.n_kv_groups, dim=1)
        t_kv = k_exp.size(2)

        use_structural_mask_only = (
            attn.momh_enabled
            and attn.momh_structural_mask_only
            and full_is_vision is not None
            and full_attention_mask is not None
            and x.device.type == "cuda"
        )

        use_soft_gating = (
            (not use_structural_mask_only)
            and attn.momh_enabled
            and attn.momh_soft_gating
            and full_is_vision is not None
            and full_attention_mask is not None
            and x.device.type == "cuda"
        )

        use_momh_modality = (
            (not use_soft_gating)
            and (not use_structural_mask_only)
            and attn.momh_enabled
            and full_is_vision is not None
            and full_attention_mask is not None
            and x.device.type == "cuda"
        )

        flex_fn = (
            flex_attention_compiled_dynamic
            if t_curr != t_kv
            else flex_attention_compiled
        )

        if use_structural_mask_only:
            block_mask = prefill_block_mask
            if block_mask is None:
                block_mask = create_base_structural_block_mask(
                    q_len=t_curr,
                    kv_len=t_kv,
                    is_vision=full_is_vision[:, :t_kv],
                    attention_mask=full_attention_mask[:, :t_kv],
                    device=str(x.device),
                )
            y = flex_fn(q, k_exp.to(q.dtype), v_exp.to(q.dtype), block_mask=block_mask)

        elif use_soft_gating:
            block_mask = prefill_block_mask
            if block_mask is None:
                if attn.momh_causal_gating:
                    block_mask = create_causal_block_mask(
                        q_len=t_curr,
                        kv_len=t_kv,
                        attention_mask=full_attention_mask[:, :t_kv],
                        device=str(x.device),
                    )
                else:
                    block_mask = create_base_structural_block_mask(
                        q_len=t_curr,
                        kv_len=t_kv,
                        is_vision=full_is_vision[:, :t_kv],
                        attention_mask=full_attention_mask[:, :t_kv],
                        device=str(x.device),
                    )

            score_mod = generate_soft_gating_score_mod(
                momh_gate=attn.momh_gate,
                is_vision=full_is_vision[:, :t_kv],
                q_offset=(t_kv - t_curr),
                active_pairs=attn.momh_soft_gating_pairs,
            )
            y = flex_fn(
                q,
                k_exp.to(q.dtype),
                v_exp.to(q.dtype),
                score_mod=score_mod,
                block_mask=block_mask,
            )

        elif use_momh_modality:
            block_mask = prefill_block_mask
            if block_mask is None:
                block_mask = create_momh_block_mask_from_modality(
                    n_q_heads=attn.n_heads,
                    q_len=t_curr,
                    kv_len=t_kv,
                    is_vision=full_is_vision[:, :t_kv],
                    attention_mask=full_attention_mask[:, :t_kv],
                    n_v_heads=attn.momh_n_v_heads,
                    n_t_heads=attn.momh_n_t_heads,
                    device=str(x.device),
                )
            y = flex_fn(q, k_exp.to(q.dtype), v_exp.to(q.dtype), block_mask=block_mask)

        else:
            additive_attn_mask = None
            if full_attention_mask is not None:
                mask_for_keys = full_attention_mask[:, :t_kv]
                additive_attn_mask = (
                    1.0 - mask_for_keys.unsqueeze(1).unsqueeze(2).to(dtype=q.dtype)
                ) * torch.finfo(q.dtype).min

            is_causal = layer_cache is None and t_curr == t_kv and t_curr > 1
            y = F.scaled_dot_product_attention(
                q,
                k_exp,
                v_exp,
                attn_mask=additive_attn_mask,
                dropout_p=0.0,
                is_causal=is_causal,
            )

        y = y.to(x.dtype)
        y = y.transpose(1, 2).contiguous().view(batch_size, t_curr, dim)
        y = attn.out_proj(y)
        y = attn.resid_dropout(y)
        return y, updated_cache

    def _forward_with_cache(
        self,
        *,
        tokens: torch.Tensor,
        full_attention_mask: torch.Tensor,
        full_is_vision: torch.Tensor,
        cache_list: list[_LayerCache | None] | None,
        start_pos: int,
        image_embd: torch.Tensor | None,
    ) -> tuple[torch.Tensor, list[_LayerCache | None]]:
        h = self.model.tok_embeddings(tokens)

        if image_embd is not None:
            vision_mask = tokens == self.model.image_token_id
            expected_vision_tokens = int(vision_mask.sum().item())
            flat_image_embd = image_embd.reshape(-1, image_embd.size(-1))
            if expected_vision_tokens != flat_image_embd.size(0):
                raise ValueError(
                    "Image embedding/token mismatch: "
                    f"expected {expected_vision_tokens}, got {flat_image_embd.size(0)}"
                )
            h = h.clone()
            h[vision_mask] = flat_image_embd.to(h.dtype)

        batch_size, t_curr = tokens.shape
        position_ids = (
            torch.arange(start_pos, start_pos + t_curr, device=tokens.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        cos, sin = self.model.rotary_embd(position_ids)

        if cache_list is None:
            cache_list = [None] * len(self.model.layers)

        first_attn = next(iter(self.model.layers.values())).attn
        kv_len = int(full_attention_mask.size(1))
        shared_block_mask = self._build_shared_block_mask(
            attn=first_attn,
            q_len=t_curr,
            kv_len=kv_len,
            full_is_vision=full_is_vision,
            full_attention_mask=full_attention_mask,
            device=tokens.device,
        )

        for idx, layer in enumerate(self.model.layers.values()):
            residual = h
            h_norm = layer.norm1(h)
            attn_out, cache_list[idx] = self._attention_with_cache(
                attn=layer.attn,
                x=h_norm,
                cos=cos,
                sin=sin,
                full_attention_mask=full_attention_mask,
                full_is_vision=full_is_vision,
                layer_cache=cache_list[idx],
                prefill_block_mask=shared_block_mask,
            )
            h = residual + attn_out

            residual = h
            h = layer.norm2(h)
            h = layer.mlp(h)
            h = residual + h

        h = self.model.norm(h)
        logits = self.model.output(h)
        return logits, cache_list

    def _sample_next_token(
        self,
        *,
        logits: torch.Tensor,
        greedy: bool,
        temperature: float,
        top_p: float,
    ) -> torch.Tensor:
        if greedy or temperature <= 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        if top_p >= 1.0:
            return torch.multinomial(probs, num_samples=1)

        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        cutoff_mask = cumulative > top_p
        cutoff_mask[:, 0] = False
        sorted_probs = sorted_probs.masked_fill(cutoff_mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
        return torch.gather(sorted_indices, dim=-1, index=sampled_idx)

    @torch.inference_mode()
    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor | None,
        max_new_tokens: int,
        greedy: bool,
        temperature: float,
        top_p: float,
    ) -> torch.Tensor:
        if max_new_tokens <= 0:
            return torch.empty(
                (input_ids.size(0), 0),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        full_attention_mask = attention_mask
        full_is_vision = input_ids == self.model.image_token_id

        image_embd = None
        if images is not None and images.numel() > 0:
            image_embd = self.model.vision_encoder(images)
            image_embd = self.model.projector(image_embd)

        prompt_len = int(input_ids.size(1))
        max_pos = int(self.model.config.lm_max_position_embeddings)
        max_new_tokens = min(max_new_tokens, max(0, max_pos - prompt_len))
        if max_new_tokens <= 0:
            return torch.empty(
                (input_ids.size(0), 0),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )

        logits, cache_list = self._forward_with_cache(
            tokens=input_ids,
            full_attention_mask=full_attention_mask,
            full_is_vision=full_is_vision,
            cache_list=None,
            start_pos=0,
            image_embd=image_embd,
        )
        current_logits = logits[:, -1, :]

        generated_tokens: list[torch.Tensor] = []
        eos_token_id = self.tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            next_token = self._sample_next_token(
                logits=current_logits,
                greedy=greedy,
                temperature=temperature,
                top_p=top_p,
            )
            generated_tokens.append(next_token)

            if eos_token_id is not None and torch.all(next_token.squeeze(-1) == eos_token_id):
                break

            full_attention_mask = torch.cat(
                (
                    full_attention_mask,
                    torch.ones_like(next_token, dtype=full_attention_mask.dtype),
                ),
                dim=1,
            )
            full_is_vision = torch.cat(
                (
                    full_is_vision,
                    torch.zeros_like(next_token, dtype=torch.bool),
                ),
                dim=1,
            )

            start_pos = int(full_attention_mask.size(1) - 1)
            decode_logits, cache_list = self._forward_with_cache(
                tokens=next_token,
                full_attention_mask=full_attention_mask,
                full_is_vision=full_is_vision,
                cache_list=cache_list,
                start_pos=start_pos,
                image_embd=None,
            )
            current_logits = decode_logits[:, -1, :]

        if not generated_tokens:
            return torch.empty(
                (input_ids.size(0), 0),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )

        return torch.cat(generated_tokens, dim=1)
