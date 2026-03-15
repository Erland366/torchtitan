"""
Mixture of Modality Heads (MoMH) Attention Module

Implements specialized attention patterns where different heads focus on different modalities:
- V-heads: Vision -> Vision only (bidirectional)
- T-heads: Text -> Text only (causal)
- VT-heads: Full cross-modal attention

Uses PyTorch's flex_attention for efficient sparse attention computation.

MoMH masking is driven by an explicit per-token modality mask (`is_vision`), typically derived
from `<|image|>` placeholder token positions.
"""

import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# Compile flex_attention for performance
# dynamic=False for prefill: fixed shapes, best performance
# dynamic=True for decode: KV length grows each step, avoid recompilation
flex_attention_compiled = torch.compile(flex_attention, dynamic=False)
flex_attention_compiled_dynamic = torch.compile(flex_attention, dynamic=True)

# Increase dynamo cache for multiple mask configurations
torch._dynamo.config.cache_size_limit = 1000

_TT_PAIR_INDEX = 0
_TV_PAIR_INDEX = 1


def get_tt_tv_pair_logits(momh_gate: torch.Tensor) -> torch.Tensor:
    """Return the active `tt`/`tv` pair logits without advanced indexing.

    DTensor currently rejects mixed advanced indexing patterns like
    ``tensor[:, [0, 1]]`` inside distributed operators. Stacking the two
    columns keeps the operation slice-based and works for both local tensors
    and DTensor-backed parameters.
    """
    if momh_gate.ndim != 2 or momh_gate.shape[1] < 2:
        raise ValueError(
            f"momh_gate must have shape [n_heads, >=2], got {tuple(momh_gate.shape)}"
        )
    return torch.stack(
        (
            momh_gate[:, _TT_PAIR_INDEX],
            momh_gate[:, _TV_PAIR_INDEX],
        ),
        dim=-1,
    )


def compute_tt_tv_balance_stats(
    momh_gate: torch.Tensor,
    *,
    target_tv: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute `tt`/`tv` balance stats from soft-gating parameters.

    Returns per-head probabilities for the `tt` and `tv` columns plus a
    differentiable mean-squared balance loss against the requested `tv` target.
    This intentionally uses only the active `tt`/`tv` slice and does not try to
    recover runtime attention mass from flex attention internals.
    """
    pair_logits = get_tt_tv_pair_logits(momh_gate)
    pair_probs = torch.softmax(pair_logits.float(), dim=-1)
    tt_prob = pair_probs[:, 0]
    tv_prob = pair_probs[:, 1]
    target = torch.full_like(tv_prob, float(target_tv))
    balance_loss = torch.mean((tv_prob - target) ** 2)
    return tt_prob, tv_prob, balance_loss


def warmup_flex_attention_compile(
    *,
    device: torch.device,
    n_heads: int,
    head_dim: int,
    include_soft_gating: bool,
) -> None:
    """Compile flex_attention call paths ahead of model block compilation.

    This warms:
    - prefill block-mask path (dynamic=False)
    - decode block-mask path (dynamic=True)
    - optional soft-gating score_mod variants for both paths
    """
    if device.type != "cuda":
        return

    # Match training dtype (bf16 on A100/Hopper) to avoid compiling only fp32 paths.
    dtype = torch.bfloat16
    B = 1
    q_len = 32
    kv_len = 32

    q_prefill = torch.randn(B, n_heads, q_len, head_dim, device=device, dtype=dtype)
    k_full = torch.randn(B, n_heads, kv_len, head_dim, device=device, dtype=dtype)
    v_full = torch.randn(B, n_heads, kv_len, head_dim, device=device, dtype=dtype)
    q_decode = torch.randn(B, n_heads, 1, head_dim, device=device, dtype=dtype)

    # Minimal but valid structural metadata for MoMH paths.
    is_vision = torch.zeros(B, kv_len, dtype=torch.bool, device=device)
    is_vision[:, : min(8, kv_len)] = True
    attention_mask = torch.ones(B, kv_len, dtype=torch.bool, device=device)

    prefill_mask = create_base_structural_block_mask(
        q_len=q_len,
        kv_len=kv_len,
        is_vision=is_vision,
        attention_mask=attention_mask,
        device=str(device),
    )
    decode_mask = create_base_structural_block_mask(
        q_len=1,
        kv_len=kv_len,
        is_vision=is_vision,
        attention_mask=attention_mask,
        device=str(device),
    )

    # Structural-only compile paths.
    _ = flex_attention_compiled(q_prefill, k_full, v_full, block_mask=prefill_mask)
    _ = flex_attention_compiled_dynamic(
        q_decode, k_full, v_full, block_mask=decode_mask
    )

    if include_soft_gating:
        # Keep gate tensors local to warmup to avoid mutating model params.
        # Gate params are stored as fp32 in training; compile score_mod with
        # fp32 gates to avoid dtype-specializing warmup to a lower-precision
        # path that does not match runtime numerics.
        momh_gate = torch.zeros(n_heads, 4, device=device, dtype=torch.float32)

        prefill_score_mod = generate_soft_gating_score_mod(
            momh_gate=momh_gate,
            is_vision=is_vision,
            q_offset=0,
            active_pairs="tt_tv",
        )
        decode_score_mod = generate_soft_gating_score_mod(
            momh_gate=momh_gate,
            is_vision=is_vision,
            q_offset=kv_len - 1,
            active_pairs="tt_tv",
        )

        _ = flex_attention_compiled(
            q_prefill,
            k_full,
            v_full,
            score_mod=prefill_score_mod,
            block_mask=prefill_mask,
        )
        _ = flex_attention_compiled_dynamic(
            q_decode,
            k_full,
            v_full,
            score_mod=decode_score_mod,
            block_mask=decode_mask,
        )

    # Prevent compiler warmup tensors from polluting peak memory tracking.
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()


def generate_momh_mask_mod_from_modality(
    n_q_heads: int,
    *,
    is_vision: torch.Tensor,
    attention_mask: torch.Tensor | None,
    q_offset: int = 0,
    n_v_heads: int,
    n_t_heads: int,
):
    """
    Generate a MoMH mask_mod based on per-token modality (vision vs text).

    Args:
        n_q_heads: Total number of query heads.
        is_vision: Bool tensor [B, KV_LEN] marking vision tokens.
        attention_mask: Optional tensor [B, KV_LEN], where 1=content and 0=padding.
        q_offset: Absolute offset to map local q_idx into KV positions.
        n_v_heads: Number of V->V heads.
        n_t_heads: Number of T->T heads.

    Returns:
        mask_mod function compatible with flex_attention's create_block_mask.
    """
    if is_vision.dtype is not torch.bool:
        is_vision = is_vision.to(torch.bool)
    if attention_mask is not None and attention_mask.dtype is not torch.bool:
        attention_mask = attention_mask.to(torch.bool)

    H_V = n_v_heads
    H_T = n_t_heads
    H_T_start = H_V
    H_VT_start = H_V + H_T

    def mask_mod(b, h, q_idx, kv_idx):
        q_abs = q_idx + q_offset
        kv_abs = kv_idx

        if attention_mask is None:
            not_padding = torch.ones_like(q_abs, dtype=torch.bool) & torch.ones_like(
                kv_abs, dtype=torch.bool
            )
        else:
            q_is_content = attention_mask[b, q_abs]
            kv_is_content = attention_mask[b, kv_abs]
            not_padding = q_is_content & kv_is_content

        q_is_vision = is_vision[b, q_abs]
        kv_is_vision = is_vision[b, kv_abs]
        q_is_text = ~q_is_vision
        kv_is_text = ~kv_is_vision

        # V-heads: V->V only (bidirectional within vision tokens)
        head_V = (h < H_T_start) & q_is_vision & kv_is_vision & not_padding

        # T-heads: T->T only (causal within text tokens)
        head_T = (
            (h >= H_T_start)
            & (h < H_VT_start)
            & q_is_text
            & kv_is_text
            & (q_abs >= kv_abs)
            & not_padding
        )

        # VT-heads: cross-modal (full vision + causal non-vision)
        head_VT = (h >= H_VT_start) & not_padding & (kv_is_vision | (q_abs >= kv_abs))

        return head_V | head_T | head_VT

    return mask_mod


def create_momh_block_mask_from_modality(
    *,
    n_q_heads: int,
    q_len: int,
    kv_len: int,
    is_vision: torch.Tensor,
    attention_mask: torch.Tensor | None,
    n_v_heads: int,
    n_t_heads: int,
    device: str = "cuda",
):
    """
    Create a MoMH BlockMask for arbitrary (Q_LEN, KV_LEN) shapes.
    """
    if is_vision.ndim != 2:
        raise ValueError(
            f"is_vision must have shape [B, KV_LEN], got {tuple(is_vision.shape)}"
        )
    if is_vision.shape[1] < kv_len:
        raise ValueError(
            f"is_vision second dim must be >= kv_len ({kv_len}), got {is_vision.shape[1]}"
        )
    if attention_mask is not None:
        if attention_mask.ndim != 2:
            raise ValueError(
                f"attention_mask must have shape [B, KV_LEN], got {tuple(attention_mask.shape)}"
            )
        if attention_mask.shape[1] < kv_len:
            raise ValueError(
                f"attention_mask second dim must be >= kv_len ({kv_len}), got {attention_mask.shape[1]}"
            )

    q_offset = kv_len - q_len
    mask_mod = generate_momh_mask_mod_from_modality(
        n_q_heads,
        is_vision=is_vision[:, :kv_len],
        attention_mask=attention_mask[:, :kv_len] if attention_mask is not None else None,
        q_offset=q_offset,
        n_v_heads=n_v_heads,
        n_t_heads=n_t_heads,
    )

    if torch.compiler.is_compiling():
        return create_block_mask(
            mask_mod,
            B=is_vision.shape[0],
            H=n_q_heads,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )

    return create_block_mask(
        mask_mod,
        B=is_vision.shape[0],
        H=n_q_heads,
        Q_LEN=q_len,
        KV_LEN=kv_len,
        device=device,
        _compile=True,
    )


def generate_base_structural_mask_mod(*, is_vision, attention_mask, q_offset=0):
    """
    Generate a head-independent mask_mod for soft-gated MoMH.

    All heads share the same structural mask:
    - Vision KV tokens are always accessible (bidirectional).
    - Text KV tokens enforce causal ordering.
    - Padding is masked out.
    """
    if is_vision.dtype is not torch.bool:
        is_vision = is_vision.to(torch.bool)
    if attention_mask is not None and attention_mask.dtype is not torch.bool:
        attention_mask = attention_mask.to(torch.bool)

    def mask_mod(b, h, q_idx, kv_idx):
        q_abs = q_idx + q_offset
        kv_is_vision = is_vision[b, kv_idx]
        is_causal = q_abs >= kv_idx
        not_padding = attention_mask[b, q_abs] & attention_mask[b, kv_idx]
        return not_padding & (kv_is_vision | is_causal)

    return mask_mod


def create_base_structural_block_mask(
    *,
    q_len: int,
    kv_len: int,
    is_vision: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str = "cuda",
):
    """
    Create a head-independent BlockMask for soft-gated MoMH.
    Uses H=None so the mask broadcasts across all heads.
    """
    if is_vision.ndim != 2:
        raise ValueError(
            f"is_vision must have shape [B, KV_LEN], got {tuple(is_vision.shape)}"
        )

    q_offset = kv_len - q_len
    mask_mod = generate_base_structural_mask_mod(
        is_vision=is_vision[:, :kv_len],
        attention_mask=attention_mask[:, :kv_len],
        q_offset=q_offset,
    )

    if torch.compiler.is_compiling():
        return create_block_mask(
            mask_mod,
            B=is_vision.shape[0],
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )

    return create_block_mask(
        mask_mod,
        B=is_vision.shape[0],
        H=None,
        Q_LEN=q_len,
        KV_LEN=kv_len,
        device=device,
        _compile=True,
    )


def create_causal_block_mask(
    *,
    q_len: int,
    kv_len: int,
    attention_mask: torch.Tensor,
    device: str = "cuda",
):
    """
    Create a head-independent causal BlockMask (no bidirectional vision).
    Uses H=None so the mask broadcasts across all heads.
    For use with momh_causal_gating: soft gates on top of standard causal attention.
    """
    if attention_mask is not None and attention_mask.dtype is not torch.bool:
        attention_mask = attention_mask.to(torch.bool)

    q_offset = kv_len - q_len

    def mask_mod(b, h, q_idx, kv_idx):
        q_abs = q_idx + q_offset
        is_causal = q_abs >= kv_idx
        not_padding = attention_mask[b, q_abs] & attention_mask[b, kv_idx]
        return not_padding & is_causal

    if torch.compiler.is_compiling():
        return create_block_mask(
            mask_mod,
            B=attention_mask.shape[0],
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )

    return create_block_mask(
        mask_mod,
        B=attention_mask.shape[0],
        H=None,
        Q_LEN=q_len,
        KV_LEN=kv_len,
        device=device,
        _compile=True,
    )


def generate_soft_gating_score_mod(
    *,
    momh_gate,
    is_vision,
    q_offset=0,
    active_pairs="all",
    gate_scale: float = 1.0,
):
    """
    Generate a score_mod that adds learnable per-head modality bias.

    The bias is indexed by (query_modality, kv_modality) pair:
        pair_idx = q_mod * 2 + kv_mod
        0=text-text, 1=text-vision, 2=vision-text, 3=vision-vision

    Args:
        momh_gate: Parameter tensor [n_heads, 4] with per-head modality biases.
        is_vision: Bool tensor [B, KV_LEN] marking vision tokens.
        q_offset: Offset to map local q_idx into absolute KV positions.
        active_pairs: "all" for all 4 pairs, "tt_tv" for text-query pairs only.
        gate_scale: Multiplier applied to the gate bias before it perturbs the
            attention score. This strengthens or weakens the realized effect of
            a given gate value without changing the underlying gate parameter.

    Returns:
        score_mod function for flex_attention.
    """
    if is_vision.dtype is not torch.bool:
        is_vision = is_vision.to(torch.bool)
    if gate_scale <= 0.0:
        raise ValueError(f"gate_scale must be > 0, got {gate_scale}")

    B_size, kv_len = is_vision.shape
    n_heads = momh_gate.shape[0]
    scaled_gate = momh_gate * float(gate_scale)

    # Precompute bias tables
    kv_mod = is_vision.to(torch.int64)  # [B, KV_LEN]

    # q=text: pair_idx = 0*2 + kv_mod -> {0=tt, 1=tv}
    pair_idx_q_text = kv_mod
    # q=vision: pair_idx = 1*2 + kv_mod -> {2=vt, 3=vv}
    pair_idx_q_vis = 2 + kv_mod

    pair_q_text_flat = (
        pair_idx_q_text.unsqueeze(0).expand(n_heads, -1, -1).reshape(n_heads, -1)
    )
    pair_q_vis_flat = (
        pair_idx_q_vis.unsqueeze(0).expand(n_heads, -1, -1).reshape(n_heads, -1)
    )

    bias_if_q_text = scaled_gate.gather(1, pair_q_text_flat).reshape(
        n_heads, B_size, kv_len
    )
    bias_if_q_vis = scaled_gate.gather(1, pair_q_vis_flat).reshape(
        n_heads, B_size, kv_len
    )

    if active_pairs == "tt_tv":
        bias_if_q_vis = torch.zeros_like(bias_if_q_vis)

    def score_mod(score, b, h, q_idx, kv_idx):
        q_abs = q_idx + q_offset
        q_is_vis = is_vision[b, q_abs]
        bias = torch.where(
            q_is_vis, bias_if_q_vis[h, b, kv_idx], bias_if_q_text[h, b, kv_idx]
        )
        return score + bias

    return score_mod
