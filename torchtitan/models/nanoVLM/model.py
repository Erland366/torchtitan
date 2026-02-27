"""
nanoVLM model for torchtitan.

Ported from nanoVLM_main/models/ — vision encoder, modality projector,
GQA decoder with MoMH attention, all in one file.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.module import Module

from .attention import (
    create_base_structural_block_mask,
    create_causal_block_mask,
    create_momh_block_mask_from_modality,
    flex_attention_compiled,
    generate_soft_gating_score_mod,
)


# ---------------------------------------------------------------------------
# Helper: build block mask outside compiled regions
# ---------------------------------------------------------------------------
@torch.compiler.disable
def _build_momh_block_mask_prefill(
    *,
    n_q_heads: int,
    seq_len: int,
    is_vision: torch.Tensor,
    attention_mask: torch.Tensor,
    n_v_heads: int,
    n_t_heads: int,
    device: str,
):
    seq_len = int(seq_len)
    return create_momh_block_mask_from_modality(
        n_q_heads=n_q_heads,
        q_len=seq_len,
        kv_len=seq_len,
        is_vision=is_vision,
        attention_mask=attention_mask,
        n_v_heads=n_v_heads,
        n_t_heads=n_t_heads,
        device=device,
    )


@torch.compiler.disable
def _build_soft_gating_block_mask_prefill(
    *,
    seq_len: int,
    is_vision: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
    causal_only: bool = False,
):
    seq_len = int(seq_len)
    if causal_only:
        return create_causal_block_mask(
            q_len=seq_len,
            kv_len=seq_len,
            attention_mask=attention_mask,
            device=device,
        )
    return create_base_structural_block_mask(
        q_len=seq_len,
        kv_len=seq_len,
        is_vision=is_vision,
        attention_mask=attention_mask,
        device=device,
    )


# ========================== Vision Encoder ==========================


class ViTPatchEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.img_size = cfg.vit_img_size
        self.patch_size = cfg.vit_patch_size
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.cls_flag = cfg.vit_cls_flag
        self.embd_dim = cfg.vit_hidden_dim

        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=self.embd_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        if self.cls_flag:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embd_dim))
            self.position_embedding = nn.Parameter(
                torch.rand(1, self.num_patches + 1, self.embd_dim)
            )
        else:
            self.position_embedding = nn.Parameter(
                torch.rand(1, self.num_patches, self.embd_dim)
            )

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        if self.cls_flag:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        x = x + self.position_embedding
        return x


class ViTMultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.vit_n_heads
        self.embd_dim = cfg.vit_hidden_dim
        assert self.embd_dim % self.n_heads == 0
        self.head_dim = self.embd_dim // self.n_heads
        self.dropout = cfg.vit_dropout

        self.qkv_proj = nn.Linear(self.embd_dim, 3 * self.embd_dim, bias=True)
        self.out_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=True)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        y = self.resid_dropout(y)
        return y


class ViTMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.activation_fn = nn.GELU(approximate="tanh")
        self.fc1 = nn.Linear(cfg.vit_hidden_dim, cfg.vit_inter_dim)
        self.fc2 = nn.Linear(cfg.vit_inter_dim, cfg.vit_hidden_dim)
        self.dropout = nn.Dropout(cfg.vit_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class ViTBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)
        self.attn = ViTMultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)
        self.mlp = ViTMLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class NanoVLMVisionEncoder(nn.Module):
    """Vision encoder using nn.ModuleDict for FSDP per-block sharding."""

    def __init__(self, cfg):
        super().__init__()
        self.patch_embedding = ViTPatchEmbeddings(cfg)
        self.cls_flag = cfg.vit_cls_flag
        self.dropout = nn.Dropout(cfg.vit_dropout)
        # Use ModuleDict (torchtitan convention for FSDP)
        self.blocks = nn.ModuleDict(
            {str(i): ViTBlock(cfg) for i in range(cfg.vit_n_blocks)}
        )
        # Use layers alias for apply_compile compatibility
        self.layers = self.blocks
        self.layer_norm = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.dropout(x)
        for block in self.blocks.values():
            x = block(x)
        if self.cls_flag:
            x = self.layer_norm(x[:, 0])
        else:
            x = self.layer_norm(x)
        return x


# ========================== Modality Projector ==========================


class ModalityProjector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_dim = cfg.vit_hidden_dim * (cfg.mp_pixel_shuffle_factor**2)
        self.output_dim = cfg.lm_hidden_dim
        self.scale_factor = cfg.mp_pixel_shuffle_factor
        self.proj = nn.Linear(self.input_dim, self.output_dim, bias=False)

    def pixel_shuffle(self, x):
        bsz, seq, embed_dim = x.size()
        seq_root = int(seq**0.5)
        assert seq_root**2 == seq
        assert seq_root % self.scale_factor == 0

        height = width = seq_root
        x = x.view(bsz, height, width, embed_dim)
        h_out = height // self.scale_factor
        w_out = width // self.scale_factor

        x = x.reshape(
            bsz, h_out, self.scale_factor, w_out, self.scale_factor, embed_dim
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(bsz, h_out * w_out, embed_dim * self.scale_factor**2)
        return x

    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = self.proj(x)
        return x


# ========================== Decoder Components ==========================


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        irms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x * irms * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=100000, max_seq_len=8192, attention_scaling=1.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.original_max_seq_len = max_seq_len
        self.attention_scaling = attention_scaling
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    @torch.no_grad()
    def forward(
        self, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = position_ids.shape
        max_seq = position_ids.max() + 1
        if max_seq > self.original_max_seq_len:
            scale = max_seq / self.original_max_seq_len
            inv_freq = self.inv_freq / scale
        else:
            inv_freq = self.inv_freq

        flat_position_ids = position_ids.reshape(-1).float()
        inv_freq = inv_freq.to(flat_position_ids.device)
        freqs = flat_position_ids.unsqueeze(-1) * inv_freq.unsqueeze(0)
        freqs = freqs.reshape(batch_size, seq_len, -1)
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = torch.cos(emb) * self.attention_scaling
        sin = torch.sin(emb) * self.attention_scaling
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_embd(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class NanoVLMGQAttention(nn.Module):
    """Grouped Query Attention with MoMH support (training-only, no KV cache)."""

    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.lm_n_heads
        self.n_kv_heads = cfg.lm_n_kv_heads
        self.embd_dim = cfg.lm_hidden_dim
        self.dropout = cfg.lm_dropout
        self.momh_enabled = getattr(cfg, "momh_enabled", False)

        # MoMH head counts from KV-group allocation
        kv_groups_vision = getattr(cfg, "momh_kv_groups_vision", None) or 0
        kv_groups_text = getattr(cfg, "momh_kv_groups_text", None) or 0
        group_size = self.n_heads // self.n_kv_heads
        self.momh_n_v_heads = kv_groups_vision * group_size
        self.momh_n_t_heads = kv_groups_text * group_size
        assert self.momh_n_v_heads + self.momh_n_t_heads <= self.n_heads

        self.momh_structural_mask_only = getattr(
            cfg, "momh_structural_mask_only", False
        )
        self.momh_causal_gating = getattr(cfg, "momh_causal_gating", False)
        self.momh_soft_gating_pairs = getattr(cfg, "momh_soft_gating_pairs", "all")
        self.momh_soft_gating = getattr(cfg, "momh_soft_gating", False)
        if self.momh_soft_gating:
            self.momh_gate = nn.Parameter(torch.zeros(self.n_heads, 4))
            if getattr(cfg, "momh_soft_gating_init", "zero") == "warm":
                with torch.no_grad():
                    NEG = -10.0
                    for h in range(self.momh_n_v_heads):
                        self.momh_gate.data[h] = torch.tensor(
                            [NEG, NEG, NEG, 0.0]
                        )
                    for h in range(
                        self.momh_n_v_heads,
                        self.momh_n_v_heads + self.momh_n_t_heads,
                    ):
                        self.momh_gate.data[h] = torch.tensor(
                            [0.0, NEG, NEG, NEG]
                        )

        assert self.n_heads % self.n_kv_heads == 0
        assert self.embd_dim % self.n_heads == 0

        self.n_kv_groups = self.n_heads // self.n_kv_heads
        self.head_dim = self.embd_dim // self.n_heads

        self.q_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=False)
        self.k_proj = nn.Linear(
            self.embd_dim, self.head_dim * self.n_kv_heads, bias=False
        )
        self.v_proj = nn.Linear(
            self.embd_dim, self.head_dim * self.n_kv_heads, bias=False
        )
        self.out_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=False)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask=None,
        block_mask=None,
        is_vision=None,
    ) -> torch.Tensor:
        B, T_curr, C = x.size()

        q = (
            self.q_proj(x)
            .view(B, T_curr, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(B, T_curr, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(B, T_curr, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        q, k = apply_rotary_pos_embd(q, k, cos, sin)

        # GQA expansion
        k_exp = k.repeat_interleave(self.n_kv_groups, dim=1)
        v_exp = v.repeat_interleave(self.n_kv_groups, dim=1)

        T_kv = k_exp.size(2)

        # Determine MoMH mode
        use_structural_mask_only = (
            self.momh_enabled
            and self.momh_structural_mask_only
            and is_vision is not None
            and attention_mask is not None
            and x.device.type == "cuda"
        )

        use_soft_gating = (
            not use_structural_mask_only
            and self.momh_enabled
            and self.momh_soft_gating
            and is_vision is not None
            and attention_mask is not None
            and x.device.type == "cuda"
        )

        use_momh_modality = (
            not use_soft_gating
            and not use_structural_mask_only
            and self.momh_enabled
            and is_vision is not None
            and attention_mask is not None
            and x.device.type == "cuda"
        )

        if use_structural_mask_only:
            if block_mask is None:
                block_mask = create_base_structural_block_mask(
                    q_len=T_curr,
                    kv_len=T_kv,
                    is_vision=is_vision[:, :T_kv],
                    attention_mask=attention_mask[:, :T_kv],
                    device=str(x.device),
                )
            target_dtype = q.dtype
            k_exp, v_exp = k_exp.to(target_dtype), v_exp.to(target_dtype)
            y = flex_attention_compiled(q, k_exp, v_exp, block_mask=block_mask)

        elif use_soft_gating:
            if block_mask is None:
                if self.momh_causal_gating:
                    block_mask = create_causal_block_mask(
                        q_len=T_curr,
                        kv_len=T_kv,
                        attention_mask=attention_mask[:, :T_kv],
                        device=str(x.device),
                    )
                else:
                    block_mask = create_base_structural_block_mask(
                        q_len=T_curr,
                        kv_len=T_kv,
                        is_vision=is_vision[:, :T_kv],
                        attention_mask=attention_mask[:, :T_kv],
                        device=str(x.device),
                    )
            score_mod = generate_soft_gating_score_mod(
                momh_gate=self.momh_gate,
                is_vision=is_vision[:, :T_kv],
                q_offset=(T_kv - T_curr),
                active_pairs=self.momh_soft_gating_pairs,
            )
            target_dtype = q.dtype
            k_exp, v_exp = k_exp.to(target_dtype), v_exp.to(target_dtype)
            y = flex_attention_compiled(
                q, k_exp, v_exp, score_mod=score_mod, block_mask=block_mask
            )

        elif use_momh_modality:
            if block_mask is None:
                block_mask = create_momh_block_mask_from_modality(
                    n_q_heads=self.n_heads,
                    q_len=T_curr,
                    kv_len=T_kv,
                    is_vision=is_vision[:, :T_kv],
                    attention_mask=attention_mask[:, :T_kv],
                    n_v_heads=self.momh_n_v_heads,
                    n_t_heads=self.momh_n_t_heads,
                    device=str(x.device),
                )
            target_dtype = q.dtype
            k_exp, v_exp = k_exp.to(target_dtype), v_exp.to(target_dtype)
            y = flex_attention_compiled(q, k_exp, v_exp, block_mask=block_mask)

        else:
            # Standard attention (no MoMH or CPU fallback)
            additive_attn_mask = None
            if attention_mask is not None:
                mask_for_keys = attention_mask[:, :T_kv]
                additive_attn_mask = (
                    1.0 - mask_for_keys.unsqueeze(1).unsqueeze(2).float()
                ) * torch.finfo(q.dtype).min

            is_causal = T_curr == T_kv and T_curr > 1
            y = F.scaled_dot_product_attention(
                q,
                k_exp,
                v_exp,
                attn_mask=additive_attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )

        y = y.to(x.dtype)
        y = y.transpose(1, 2).contiguous().view(B, T_curr, C)
        y = self.out_proj(y)
        y = self.resid_dropout(y)
        return y


class NanoVLMFeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.lm_hidden_dim, cfg.lm_inter_dim, bias=False)
        self.up_proj = nn.Linear(cfg.lm_hidden_dim, cfg.lm_inter_dim, bias=False)
        self.down_proj = nn.Linear(cfg.lm_inter_dim, cfg.lm_hidden_dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class NanoVLMDecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp = NanoVLMFeedForward(cfg)
        self.attn = NanoVLMGQAttention(cfg)
        self.norm1 = RMSNorm(cfg.lm_hidden_dim, eps=cfg.lm_rms_eps)
        self.norm2 = RMSNorm(cfg.lm_hidden_dim, eps=cfg.lm_rms_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask=None,
        block_mask=None,
        is_vision=None,
    ):
        res = x
        x = self.norm1(x)
        x = self.attn(
            x,
            cos,
            sin,
            attention_mask=attention_mask,
            block_mask=block_mask,
            is_vision=is_vision,
        )
        x = res + x

        res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = res + x
        return x


# ========================== Top-level VLM ==========================


class NanoVLMModel(BaseModel):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        # Vision
        vit_hidden_dim: int = 768
        vit_inter_dim: int = 3072
        vit_patch_size: int = 16
        vit_img_size: int = 512
        vit_n_heads: int = 12
        vit_n_blocks: int = 12
        vit_ln_eps: float = 1e-6
        vit_cls_flag: bool = False
        vit_dropout: float = 0.0

        # Language
        lm_hidden_dim: int = 960
        lm_inter_dim: int = 2560
        lm_rms_eps: float = 1e-5
        lm_re_base: int = 100000
        lm_max_position_embeddings: int = 8192
        lm_vocab_size: int = 49218
        lm_n_heads: int = 15
        lm_n_kv_heads: int = 5
        lm_n_blocks: int = 32
        lm_attn_scaling: float = 1.0
        lm_tie_weights: bool = True
        lm_dropout: float = 0.0

        # Projector
        mp_pixel_shuffle_factor: int = 4
        mp_image_token_length: int = 64

        # MoMH
        momh_enabled: bool = False
        momh_kv_groups_vision: int | None = None
        momh_kv_groups_text: int | None = None
        momh_soft_gating: bool = False
        momh_soft_gating_init: str = "zero"
        momh_soft_gating_pairs: str = "all"
        momh_structural_mask_only: bool = False
        momh_causal_gating: bool = False

        # Identity (set during build from tokenizer)
        image_token_id: int = -1

        def update_from_config(self, *, trainer_config, **kwargs) -> None:
            # Pull image_token_id from the dataloader config if available
            dl_config = getattr(trainer_config, "dataloader", None)
            if dl_config is not None and hasattr(dl_config, "image_token_id"):
                if dl_config.image_token_id > 0:
                    self.image_token_id = dl_config.image_token_id

        def get_nparams_and_flops(
            self, model: Module, seq_len: int
        ) -> tuple[int, int]:
            nparams = sum(p.numel() for p in model.parameters())
            # Rough FLOPs estimate: 6 * nparams * seq_len (standard transformer approx)
            flops = 6 * nparams * seq_len
            return nparams, flops

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.image_token_id = config.image_token_id

        # Vision encoder
        self.vision_encoder = NanoVLMVisionEncoder(config)

        # Modality projector
        self.projector = ModalityProjector(config)

        # Token embeddings
        self.tok_embeddings = nn.Embedding(config.lm_vocab_size, config.lm_hidden_dim)

        # Rotary embeddings
        head_dim = config.lm_hidden_dim // config.lm_n_heads
        self.rotary_embd = RotaryEmbedding(
            dim=head_dim,
            base=config.lm_re_base,
            max_seq_len=config.lm_max_position_embeddings,
            attention_scaling=config.lm_attn_scaling,
        )

        # Decoder layers (ModuleDict for FSDP)
        self.layers = nn.ModuleDict(
            {str(i): NanoVLMDecoderBlock(config) for i in range(config.lm_n_blocks)}
        )

        # Final norm + LM head
        self.norm = RMSNorm(config.lm_hidden_dim, eps=config.lm_rms_eps)
        self.output = nn.Linear(config.lm_hidden_dim, config.lm_vocab_size, bias=False)

        # Weight tying
        if config.lm_tie_weights:
            self.output.weight = self.tok_embeddings.weight

    def init_weights(self, **kwargs):
        # Vision encoder
        for module in self.vision_encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # ViT position embedding and CLS token (raw Parameters)
        patch_emb = self.vision_encoder.patch_embedding
        nn.init.uniform_(patch_emb.position_embedding, -1.0, 1.0)
        if patch_emb.cls_flag:
            nn.init.zeros_(patch_emb.cls_token)

        # Modality projector
        for module in self.projector.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Token embeddings
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=0.02)

        # Rotary embedding inv_freq buffer
        dim = self.rotary_embd.dim
        base = self.rotary_embd.base
        device = kwargs.get("buffer_device", self.rotary_embd.inv_freq.device)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, device=device).float() / dim)
        )
        self.rotary_embd.inv_freq = inv_freq

        # Decoder layers
        for layer in self.layers.values():
            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, RMSNorm):
                    module.weight.data.fill_(1.0)

        # Final norm
        self.norm.weight.data.fill_(1.0)

        # Output head: re-tie or init
        if self.config.lm_tie_weights:
            self.output.weight = self.tok_embeddings.weight
        else:
            nn.init.normal_(self.output.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        images: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for nanoVLM.

        Args:
            tokens: [B, seq_len] token IDs
            images: [N, C, H, W] image tensors (from extra_inputs)
            attention_mask: [B, seq_len] attention mask (from extra_inputs)
            **kwargs: absorb extra_kwargs from trainer

        Returns:
            logits: [B, seq_len, vocab_size]
        """
        is_vision = tokens == self.image_token_id

        # Build MoMH block mask once (outside compiled blocks)
        prefill_block_mask = None
        if (
            attention_mask is not None
            and tokens.device.type == "cuda"
            and self.config.momh_enabled
            and tokens.size(1) > 1
        ):
            first_block = next(iter(self.layers.values()))
            attn_obj = first_block.attn
            seq_len = int(tokens.size(1))

            if attn_obj.momh_soft_gating or attn_obj.momh_structural_mask_only:
                prefill_block_mask = _build_soft_gating_block_mask_prefill(
                    seq_len=seq_len,
                    is_vision=is_vision[:, :seq_len],
                    attention_mask=attention_mask[:, :seq_len],
                    device=str(tokens.device),
                    causal_only=attn_obj.momh_causal_gating,
                )
            else:
                prefill_block_mask = _build_momh_block_mask_prefill(
                    n_q_heads=int(attn_obj.n_heads),
                    seq_len=seq_len,
                    is_vision=is_vision[:, :seq_len],
                    attention_mask=attention_mask[:, :seq_len],
                    n_v_heads=int(attn_obj.momh_n_v_heads),
                    n_t_heads=int(attn_obj.momh_n_t_heads),
                    device=str(tokens.device),
                )

        # Token embeddings
        h = self.tok_embeddings(tokens)

        # Vision processing: images must be pre-collated as a single tensor.
        if isinstance(images, list):
            raise TypeError("Expected `images` to be a tensor, got list")
        if images is not None and images.numel() > 0:
            image_embd = self.vision_encoder(images)
            image_embd = self.projector(image_embd)
            # Replace image token placeholders with vision embeddings
            mask = is_vision
            h = h.clone()
            h[mask] = image_embd.reshape(-1, image_embd.size(-1)).to(h.dtype)

        # Compute RoPE
        B, T_curr = tokens.shape
        position_ids = torch.arange(T_curr, device=tokens.device).unsqueeze(0).expand(B, -1)
        cos, sin = self.rotary_embd(position_ids)

        # Decoder layers
        for layer in self.layers.values():
            h = layer(
                h,
                cos,
                sin,
                attention_mask=attention_mask,
                block_mask=prefill_block_mask,
                is_vision=is_vision,
            )

        # Final norm + LM head
        h = self.norm(h)
        logits = self.output(h)
        return logits
