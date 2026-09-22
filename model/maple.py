"""
MAPLE: Modality-Adaptive Prior-Guided Learning with Language-Conditioned
       Spatial Adaptation for Multimodal Remote Sensing Segmentation.

Module naming follows the paper:
    LCAS  - Language-Conditioned Adapter Synthesis        (Sec. III-B, Eq. 2-3)
    LMA   - Language-conditioned Modality-specific Adapter (Sec. III-B, Eq. 1)
    VLP   - Vision-Language Prior                          (Sec. III-C, Eq. 4-8)
    DCH   - Dense Classification Head                      (Sec. III-C, Eq. 7)
    SAFR  - Spatial Adaptive Frequency Reweighting         (Sec. III-D, Eq. 9-15)
"""
import math
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath, trunc_normal_
import open_clip

from sam3.model.vitdet import ViT, window_partition, window_unpartition

from .base import BaseSegmentor
from src.registries import MODEL_REGISTRY


# ============================================================
#  LCAS:  Language-Conditioned Adapter Synthesis  (Sec. III-B)
# ============================================================

class LCAS(nn.Module):
    """
    Generates spatially adaptive, modality-aware FiLM parameters
    (s_tilde, h_tilde) from the VLP similarity map  (Eq. 2-3).

    Three-level conditioning:
        - Semantic : per-class bases  s_c, h_c          (one set, shared)
        - Spatial  : softmax(S_sim / T) -> w_c(i, j)     (Eq. 2)
        - Modality : gates  g^(m) = sigmoid(W_g e_m)*2   (per modality)

    Note on layer selection:
        LCAS does not need to condition every transformer block. Empirically,
        injecting RemoteCLIP semantics into shallow layers (which mainly encode
        edges/textures) brings little benefit and adds noise. We therefore
        restrict conditioning to a small set of `lcas_layers` (e.g. SAM3's
        own global-attention layers (7, 15, 23, 31)). Non-conditioned layers
        keep a plain Houlsby-style LMA bottleneck (see LMALayer mode (b)).
        This also reduces the per-class base parameters from
        num_blocks * C * r  down to  len(lcas_layers) * C * r.
    """

    def __init__(self, num_classes, num_lcas_layers, adapter_bottleneck_dim,
                 num_modalities=2):
        super().__init__()
        self.num_lcas_layers = num_lcas_layers
        self.num_classes = num_classes
        self.dim = adapter_bottleneck_dim
        self.num_modalities = num_modalities

        # Per-class scale / shift bases  {s_c^(l), h_c^(l)}  (Eq. 2)
        # Only stored for layers that actually use LCAS.
        self.class_scales_attn = nn.Parameter(
            torch.zeros(num_lcas_layers, num_classes, adapter_bottleneck_dim))
        self.class_shifts_attn = nn.Parameter(
            torch.zeros(num_lcas_layers, num_classes, adapter_bottleneck_dim))
        self.class_scales_mlp = nn.Parameter(
            torch.zeros(num_lcas_layers, num_classes, adapter_bottleneck_dim))
        self.class_shifts_mlp = nn.Parameter(
            torch.zeros(num_lcas_layers, num_classes, adapter_bottleneck_dim))

        # Modality embeddings {e_0, e_1} and gating projections W_g
        self.modality_embeddings = nn.Parameter(
            torch.randn(num_modalities, adapter_bottleneck_dim) * 0.02)
        self.gate_scale_attn = nn.Linear(adapter_bottleneck_dim, adapter_bottleneck_dim)
        self.gate_shift_attn = nn.Linear(adapter_bottleneck_dim, adapter_bottleneck_dim)
        self.gate_scale_mlp = nn.Linear(adapter_bottleneck_dim, adapter_bottleneck_dim)
        self.gate_shift_mlp = nn.Linear(adapter_bottleneck_dim, adapter_bottleneck_dim)

        # Temperature T for the spatial softmax in Eq. 2
        self.temperature = nn.Parameter(torch.ones(1))

        # Identity-gate init: sigmoid(0) * 2 = 1
        for gate in [self.gate_scale_attn, self.gate_shift_attn,
                     self.gate_scale_mlp, self.gate_shift_mlp]:
            nn.init.constant_(gate.weight, 0.0)
            nn.init.constant_(gate.bias, 0.0)

    def forward(self, sim_map, modality_id):
        """
        Args:
            sim_map: S_sim, [B, num_classes, H_clip, W_clip]
            modality_id: 0 = RGB, 1 = DSM
        Returns:
            film_params_attn / film_params_mlp:
                lists of length num_lcas_layers, each entry (s_c, h_c)
            spatial_weights: w_c(i, j)
        """
        # Spatial weights  w_c(i, j) = softmax(S_sim / T)   (Eq. 2)
        spatial_weights = F.softmax(
            sim_map / self.temperature.clamp(min=0.01), dim=1)

        # Modality gate  g^(m) = sigmoid(W_g e_m) * 2
        m_emb = self.modality_embeddings[modality_id]
        g_sa = torch.sigmoid(self.gate_scale_attn(m_emb)) * 2.0
        g_ha = torch.sigmoid(self.gate_shift_attn(m_emb)) * 2.0
        g_sm = torch.sigmoid(self.gate_scale_mlp(m_emb)) * 2.0
        g_hm = torch.sigmoid(self.gate_shift_mlp(m_emb)) * 2.0

        # Per-LCAS-layer (semantic ⊗ modality) bases; spatial mixing in LMALayer
        film_params_attn, film_params_mlp = [], []
        for i in range(self.num_lcas_layers):
            s_a = torch.sigmoid(self.class_scales_attn[i]) * 2.0 * g_sa
            h_a = self.class_shifts_attn[i] * g_ha
            film_params_attn.append((s_a, h_a))

            s_m = torch.sigmoid(self.class_scales_mlp[i]) * 2.0 * g_sm
            h_m = self.class_shifts_mlp[i] * g_hm
            film_params_mlp.append((s_m, h_m))

        return film_params_attn, film_params_mlp, spatial_weights


# ============================================================
#  LMA:  Language-conditioned Modality-specific Adapters
#        (Sec. III-B, Eq. 1)
# ============================================================

class LMALayer(nn.Module):
    """
    Bottleneck adapter that operates in two modes:

      (a) LCAS-conditioned (when class_scales is provided), realising Eq. 1:
            x' = x + alpha * W_up( s_tilde(i,j) ⊙ GELU(W_down x) + h_tilde(i,j) )
          where (s_tilde, h_tilde) are obtained by spatially mixing per-class
          bases (s_c, h_c) supplied by LCAS with weights w_c(i, j) (Eq. 2).

      (b) Plain Houlsby-style adapter (when class_scales is None):
            x' = x + alpha * W_up( GELU(W_down x) )
          Used in layers that are not selected for LCAS conditioning, where
          spatial / semantic modulation provides no useful signal but the
          bottleneck still performs lightweight modality adaptation.
    """

    def __init__(self, dim, reduction=32, init_scale=0.1):
        super().__init__()
        self.down = nn.Linear(dim, reduction)        # W_down
        self.act = nn.GELU()
        self.up = nn.Linear(reduction, dim)          # W_up
        self.scale = nn.Parameter(torch.ones(1) * init_scale)   # alpha

        trunc_normal_(self.down.weight, std=.02)
        nn.init.constant_(self.down.bias, 0)
        trunc_normal_(self.up.weight, std=.02)
        nn.init.constant_(self.up.bias, 0)

    def forward(self, x, class_scales=None, class_shifts=None,
                spatial_weights=None):
        """
        x:               [B, H, W, dim] or [B, N, dim]
        class_scales:    [num_classes, reduction]   = s_c    (or None)
        class_shifts:    [num_classes, reduction]   = h_c    (or None)
        spatial_weights: [B, num_classes, H', W']   = w_c    (or None)
        """
        input_3d = (x.ndim == 3)
        if input_3d:
            B, N, D = x.shape
            H = W = int(math.sqrt(N))
            x = x.reshape(B, H, W, D)

        B, H, W, D = x.shape
        h = self.act(self.down(x))   # GELU(W_down x)

        if class_scales is not None:
            # LCAS-conditioned: spatially mix per-class bases (Eq. 2)
            w = F.interpolate(spatial_weights, size=(H, W),
                              mode='bilinear', align_corners=False)
            s_tilde = torch.einsum('bchw,cr->bhwr', w, class_scales)
            h_tilde = torch.einsum('bchw,cr->bhwr', w, class_shifts)
            h = s_tilde * h + h_tilde
        # else: plain bottleneck adapter (no FiLM mixing)

        out = self.scale * self.up(h)
        if input_3d:
            out = out.reshape(B, -1, out.shape[-1])
        return out


class LMA(nn.Module):
    """
    Container for one modality's set of LMA layers (Phi_rgb or Phi_dsm).
    Two adapters per transformer block: one after self-attention, one after FFN.
    """

    def __init__(self, num_blocks, dim, reduction=32):
        super().__init__()
        self.lma_attn = nn.ModuleList([
            LMALayer(dim, reduction) for _ in range(num_blocks)])
        self.lma_mlp = nn.ModuleList([
            LMALayer(dim, reduction) for _ in range(num_blocks)])

    def get_lma(self, block_idx):
        return self.lma_attn[block_idx], self.lma_mlp[block_idx]


# ============================================================
#  SAM3 ViT forward with LMA injection
# ============================================================

def forward_block_with_lma(block, x, lma_attn, lma_mlp,
                           cs_attn, ch_attn, cs_mlp, ch_mlp,
                           spatial_weights):
    shortcut = x
    x = block.norm1(x)

    if block.window_size > 0:
        H, W = x.shape[1], x.shape[2]
        x, pad_hw = window_partition(x, block.window_size)

    x = block.ls1(block.attn(x))

    if block.window_size > 0:
        x = window_unpartition(x, block.window_size, pad_hw, (H, W))

    x = shortcut + block.drop_path(x)
    x = x + lma_attn(x, cs_attn, ch_attn, spatial_weights)

    x = x + block.drop_path(block.ls2(block.mlp(block.norm2(x))))
    x = x + lma_mlp(x, cs_mlp, ch_mlp, spatial_weights)
    return x


def forward_vit_with_lma(vit, x, lma, lcas_layers,
                         film_params_attn, film_params_mlp,
                         spatial_weights):
    """
    Args:
        lcas_layers: sorted list of block indices (in vit.blocks) that
            consume LCAS conditioning. Other blocks fall back to plain
            bottleneck adapters.
        film_params_attn / film_params_mlp:
            lists of length len(lcas_layers); film_params_attn[k] holds the
            (s_c, h_c) tuple for block lcas_layers[k].
    """
    from sam3.model.vitdet import get_abs_pos

    x = vit.patch_embed(x)
    h, w = x.shape[1], x.shape[2]

    s = 0
    if vit.retain_cls_token:
        x = torch.cat([vit.class_embedding, x.flatten(1, 2)], dim=1)
        s = 1

    if vit.pos_embed is not None:
        x = x + get_abs_pos(
            vit.pos_embed, vit.pretrain_use_cls_token,
            (h, w), vit.retain_cls_token, tiling=vit.tile_abs_pos)

    x = vit.ln_pre(x)

    # block_idx -> position in film_params lists
    lcas_lookup = {b: k for k, b in enumerate(lcas_layers)}

    outputs = []
    for i, blk in enumerate(vit.blocks):
        lma_attn, lma_mlp = lma.get_lma(i)

        if i in lcas_lookup:
            k = lcas_lookup[i]
            cs_attn, ch_attn = film_params_attn[k]
            cs_mlp, ch_mlp = film_params_mlp[k]
            sw = spatial_weights
        else:
            cs_attn = ch_attn = cs_mlp = ch_mlp = None
            sw = None

        if vit.use_act_checkpoint and vit.training:
            import torch.utils.checkpoint as checkpoint
            x = checkpoint.checkpoint(
                forward_block_with_lma,
                blk, x, lma_attn, lma_mlp,
                cs_attn, ch_attn, cs_mlp, ch_mlp, sw,
                use_reentrant=False)
        else:
            x = forward_block_with_lma(
                blk, x, lma_attn, lma_mlp,
                cs_attn, ch_attn, cs_mlp, ch_mlp, sw)

        if (i == vit.full_attn_ids[-1]) or \
           (vit.return_interm_layers and i in vit.full_attn_ids):
            if i == vit.full_attn_ids[-1]:
                x = vit.ln_post(x)
            feats = x[:, s:]
            if feats.ndim == 4:
                feats = feats.permute(0, 3, 1, 2)
            else:
                h_out = w_out = int(math.sqrt(feats.shape[1]))
                feats = feats.reshape(
                    feats.shape[0], h_out, w_out, feats.shape[-1]
                ).permute(0, 3, 1, 2)
            outputs.append(feats)

    return outputs


def create_sam3_vit(img_size=336):
    """SAM3 Perception Encoder (ViT-L+) configuration."""
    return ViT(
        img_size=img_size, pretrain_img_size=336, patch_size=14,
        embed_dim=1024, depth=32, num_heads=16, mlp_ratio=4.625,
        norm_layer="LayerNorm", drop_path_rate=0.1, qkv_bias=True,
        use_abs_pos=True, tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31), rel_pos_blocks=(),
        use_rope=True, use_interp_rope=True, window_size=24,
        pretrain_use_cls_token=True, retain_cls_token=False,
        ln_pre=True, ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False, compile_mode=None,
    )


def load_sam3_weights(vit, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    new_ckpt = {}
    for k, v in ckpt.items():
        if "detector.backbone.vision_backbone.trunk" in k and 'freqs_cis' not in k:
            new_ckpt[k[len("detector.backbone.vision_backbone.trunk."):]] = v
    vit.load_state_dict(new_ckpt, strict=False)


# ============================================================
#  Cross-Modal Fusion (scale-wise SE fusion)
# ============================================================

class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1),
            nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(F.adaptive_avg_pool2d(x, 1))


class SEFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.se_rgb = SqueezeAndExcitation(channels)
        self.se_dsm = SqueezeAndExcitation(channels)

    def forward(self, rgb, dsm):
        return self.se_rgb(rgb) + self.se_dsm(dsm)


# ============================================================
#  VLP:  Vision-Language Prior  (Sec. III-C)
# ============================================================

class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        return F.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        ).to(x.dtype)


class DCH(nn.Module):
    """
    Dense Classification Head (Eq. 7):
        z_hat_i = z_i + MLP(LN(z_i))
        Y_dense(i) = W_cls · LN(z_hat_i)
    """

    def __init__(self, width, layers=1, mlp_ratio=4.0, act_layer=nn.GELU,
                 norm_layer=LayerNorm, output_dim=6):
        super().__init__()
        self.width = width
        mlp_width = int(width * mlp_ratio)
        self.mlps = nn.ModuleList([nn.Sequential(OrderedDict([
            ("c_norm", norm_layer(width)),
            ("c_fc", nn.Linear(width, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, width))
        ])) for _ in range(layers)])
        self.ln_mlp = norm_layer(width)
        self.projection = nn.Linear(width, output_dim)        # W_cls

        proj_std = (width ** -0.5) * (2 ** -0.5)
        fc_std = (2 * width) ** -0.5
        for b in self.mlps:
            nn.init.normal_(b.c_fc.weight, std=fc_std)
            nn.init.normal_(b.c_proj.weight, std=proj_std)
        nn.init.normal_(self.projection.weight, std=width ** -0.5)
        self.projection.bias.data.fill_(0.0)

    def forward(self, z):
        for mlp in self.mlps:
            z = z + mlp(z)                        # z_hat
        return self.projection(self.ln_mlp(z))    # Y_dense


class VLPAdapter(nn.Module):
    """Bottleneck adapter inserted into each frozen RemoteCLIP block.

    Same residual design as Eq. 1 but without spatial / modality conditioning.
    """

    def __init__(self, dim, reduction=4):
        super().__init__()
        h = dim // reduction
        self.adapter = nn.Sequential(
            nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        return x + self.scale * self.adapter(x)


def encode_text_with_prompt_ensemble(model, tokenizer, classes, device='cuda'):
    """Class embedding f_c^t via prompt ensembling (Eq. 4)."""
    templates = [
        "a satellite image of {}", "an aerial view of {}",
        "a remote sensing image of {}", "a top-down view of {}",
        "{} in a satellite image", "satellite imagery showing {}",
    ]
    feats = []
    with torch.no_grad():
        for cls in classes:
            toks = tokenizer([t.format(cls) for t in templates]).to(device)
            tf = model.encode_text(toks).float()
            tf = tf / tf.norm(dim=-1, keepdim=True)
            tf = tf.mean(0, keepdim=True)
            tf = tf / tf.norm(dim=-1, keepdim=True)
            feats.append(tf)
    return torch.cat(feats, dim=0)


class VLP(nn.Module):
    """
    Vision-Language Prior module (Sec. III-C):
        - Frozen RemoteCLIP backbone with VLPAdapter inserted per block.
        - Similarity branch  -> S_sim       (Eq. 5-6, semantic prior).
        - DCH branch         -> Y_dense     (Eq. 7,   dense prior).
    """

    def __init__(self, model_name, checkpoint_path, device='cuda',
                 num_classes=6, adapter_reduction=4,
                 dch_layers=1, dch_mlp_ratio=4.0):
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.num_classes = num_classes

        # Load OpenCLIP architecture and RemoteCLIP weights
        self.clip_model, _, _ = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.clip_model.load_state_dict(ckpt)
        self.clip_model = self.clip_model.to(device).float()

        # Backbone-specific dimensions
        self.input_resolution = 224
        if 'ViT-L-14' in model_name:
            self.feature_dim, self.hidden_dim, self.patch_size = 768, 1024, 14
        else:                                        # default ViT-B/32
            self.feature_dim, self.hidden_dim, self.patch_size = 512, 768, 32
        self.grid_size = self.input_resolution // self.patch_size
        self.num_layers = len(self.clip_model.visual.transformer.resblocks)

        # Freeze backbone, insert lightweight VLP adapters
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.vlp_adapters = nn.ModuleList([
            VLPAdapter(self.hidden_dim, adapter_reduction)
            for _ in range(self.num_layers)])

        # Dense Classification Head (Eq. 7)
        self.dch = DCH(self.hidden_dim, dch_layers, dch_mlp_ratio,
                       output_dim=num_classes)

    def encode_image_with_adapter(self, images):
        vis = self.clip_model.visual
        x = vis.conv1(images.float())
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = vis.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1) + vis.positional_embedding.to(x.dtype)
        x = vis.ln_pre(x).permute(1, 0, 2)
        for i, blk in enumerate(vis.transformer.resblocks):
            x = blk(x)
            x = x.permute(1, 0, 2)
            x = self.vlp_adapters[i](x)
            x = x.permute(1, 0, 2)
        x = x.permute(1, 0, 2)
        return x[:, 1:, :], x[:, 0, :]                 # patch tokens z_i, cls

    def forward(self, images, text_prompts):
        """
        Returns:
            'sim_map'    : S_sim   [B, C, H', W']      (Eq. 5-6)
            'dense_logits': Y_dense [B, C, H', W']      (Eq. 7)
            'patch_tokens': z_i
            'text_features': f_c^t
        """
        device = images.device
        B = images.shape[0]
        clip_img = F.interpolate(
            images, size=(self.input_resolution,) * 2,
            mode='bilinear', align_corners=False)

        z, _ = self.encode_image_with_adapter(clip_img)

        # Visual projection f_v_tilde = W_proj · LN(z)         (Eq. 5)
        vis = self.clip_model.visual
        f_v = vis.ln_post(z)
        if vis.proj is not None:
            f_v = f_v @ vis.proj
        f_v = f_v / f_v.norm(dim=-1, keepdim=True)

        # Class text embeddings f_c^t                          (Eq. 4)
        f_t = encode_text_with_prompt_ensemble(
            self.clip_model, self.tokenizer, text_prompts, device)

        # Similarity prior S_sim (cosine + min-max normalize)   (Eq. 5-6)
        sim = torch.einsum('bnd,cd->bnc', f_v, f_t)
        sim = sim.permute(0, 2, 1).reshape(B, -1, self.grid_size, self.grid_size)
        smin = sim.flatten(1).min(1, keepdim=True)[0][..., None, None]
        smax = sim.flatten(1).max(1, keepdim=True)[0][..., None, None]
        sim_map = (sim - smin) / (smax - smin + 1e-8)

        # Dense prior Y_dense                                   (Eq. 7)
        dense_logits = self.dch(z).permute(0, 2, 1).reshape(
            B, -1, self.grid_size, self.grid_size)

        return {
            'sim_map': sim_map,
            'dense_logits': dense_logits,
            'patch_tokens': z,
            'text_features': f_t,
        }


# ============================================================
#  SAFR:  Spatial Adaptive Frequency Reweighting   (Sec. III-D)
# ============================================================

class SAFRLoss(nn.Module):
    """
    Auxiliary loss applied to DCH logits with online spatial-frequency-based
    class reweighting (Eq. 9-15).

        Stage-1 EMA (beta1):   f_c     <- beta1 f_c^(t-1) + (1-beta1) f_hat_c
        ISF + min-max norm:    w_tilde_c in [0.5, 1.5]
        Stage-2 EMA (beta2):   w_c     <- beta2 w_c^(t-1) + (1-beta2) w_tilde_c
        Loss:                  weighted-mean cross-entropy over patches
    """

    def __init__(self, num_classes=6, grid_size=7, loss_weight=0.1,
                 ignore_index=255, beta1=0.99, beta2=0.9, warmup_steps=100):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.loss_weight = loss_weight              # lambda  (Eq. 15)
        self.ignore_index = ignore_index
        self.beta1 = beta1                          # Eq. 10
        self.beta2 = beta2                          # Eq. 13
        self.warmup_steps = warmup_steps            # T_warm

        # f_c^(t)   smoothed empirical class frequency
        self.register_buffer("class_freq",
                             torch.zeros(num_classes, dtype=torch.float64))
        # iteration counter t
        self.register_buffer("num_samples",
                             torch.zeros(1, dtype=torch.float64))
        # w_c^(t)   final class weight applied to CE
        self.register_buffer("class_weights",
                             torch.ones(num_classes, dtype=torch.float32))

    def forward(self, vlp_output, gt_mask):
        logits = vlp_output['dense_logits']          # z^(i)
        # y_i^down  - GT downsampled to patch grid
        gt = F.interpolate(
            gt_mask.unsqueeze(1).float(),
            size=(self.grid_size,) * 2, mode='nearest'
        ).squeeze(1).long()

        # Stage-1: empirical frequency f_hat_c  (Eq. 9)  -> EMA  (Eq. 10)
        if self.training:
            B = gt.shape[0]
            total = self.grid_size ** 2
            f_hat = torch.stack([
                (gt == c).float().sum() / (B * total)
                for c in range(self.num_classes)])
            if self.num_samples < 1:
                self.class_freq.copy_(f_hat)
            else:
                self.class_freq.mul_(self.beta1).add_(f_hat * (1 - self.beta1))
            self.num_samples += 1

        # Stage-2: ISF (Eq. 11) -> normalize to [0.5, 1.5] (Eq. 12) -> EMA (Eq. 13)
        w = None
        if self.training and self.num_samples > self.warmup_steps:
            eps = 1e-6
            isf = torch.log((1 + eps) / (self.class_freq + eps))
            isf = (isf - isf.min()) / (isf.max() + eps)         # [0, 1]
            isf = (0.5 + isf).float().to(logits.device)         # [0.5, 1.5]
            self.class_weights.mul_(self.beta2).add_(isf * (1 - self.beta2))
            w = self.class_weights.clone()

        # Eq. 14: weighted-mean cross-entropy (PyTorch CE w/ weight uses this form)
        return self.loss_weight * F.cross_entropy(
            logits, gt, weight=w, ignore_index=self.ignore_index)


# ============================================================
#  UNetFormer-style Decoder (with VLP prior fusion, Eq. 8)
# ============================================================

class ConvBNReLU(nn.Sequential):
    def __init__(self, inc, outc, ks=3, d=1, s=1, norm=nn.BatchNorm2d, bias=False):
        pad = ((s - 1) + d * (ks - 1)) // 2
        super().__init__(
            nn.Conv2d(inc, outc, ks, s, pad, d, bias=bias),
            norm(outc), nn.ReLU6())


class ConvBN(nn.Sequential):
    def __init__(self, inc, outc, ks=3, d=1, s=1, norm=nn.BatchNorm2d, bias=False):
        pad = ((s - 1) + d * (ks - 1)) // 2
        super().__init__(
            nn.Conv2d(inc, outc, ks, s, pad, d, bias=bias), norm(outc))


class Conv(nn.Sequential):
    def __init__(self, inc, outc, ks=3, d=1, s=1, bias=False):
        pad = ((s - 1) + d * (ks - 1)) // 2
        super().__init__(nn.Conv2d(inc, outc, ks, s, pad, d, bias=bias))


class SeparableConvBN(nn.Sequential):
    def __init__(self, inc, outc, ks=3, s=1, d=1, norm=nn.BatchNorm2d):
        pad = ((s - 1) + d * (ks - 1)) // 2
        super().__init__(
            nn.Conv2d(inc, inc, ks, s, pad, d, groups=inc, bias=False),
            norm(inc),
            nn.Conv2d(inc, outc, 1, bias=False))


class Mlp(nn.Module):
    def __init__(self, inf, hf=None, of=None, act=nn.ReLU6, drop=0.):
        super().__init__()
        of = of or inf
        hf = hf or inf
        self.fc1 = nn.Conv2d(inf, hf, 1, bias=True)
        self.act = act()
        self.fc2 = nn.Conv2d(hf, of, 1, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class GlobalLocalAttention(nn.Module):
    def __init__(self, dim=256, nh=16, qkv_bias=False, ws=8, rpe=True):
        super().__init__()
        self.num_heads = nh
        self.scale = (dim // nh) ** -0.5
        self.ws = ws
        self.qkv = Conv(dim, 3 * dim, 1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, 3)
        self.local2 = ConvBN(dim, dim, 1)
        self.proj = SeparableConvBN(dim, dim, ws)
        self.attn_x = nn.AvgPool2d((ws, 1), 1, (ws // 2 - 1, 0))
        self.attn_y = nn.AvgPool2d((1, ws), 1, (0, ws // 2 - 1))
        self.rpe = rpe
        if rpe:
            self.rpb_table = nn.Parameter(
                torch.zeros((2 * ws - 1) * (2 * ws - 1), nh))
            ch, cw = torch.arange(ws), torch.arange(ws)
            co = torch.stack(torch.meshgrid([ch, cw], indexing='ij'))
            cf = torch.flatten(co, 1)
            rc = cf[:, :, None] - cf[:, None, :]
            rc = rc.permute(1, 2, 0).contiguous()
            rc[:, :, 0] += ws - 1
            rc[:, :, 1] += ws - 1
            rc[:, :, 0] *= 2 * ws - 1
            self.register_buffer("rpi", rc.sum(-1))
            trunc_normal_(self.rpb_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        local = self.local2(x) + self.local1(x)
        x = self.pad(x, self.ws)
        _, _, Hp, Wp = x.shape
        q, k, v = rearrange(
            self.qkv(x),
            'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d',
            h=self.num_heads, d=C // self.num_heads,
            hh=Hp // self.ws, ww=Wp // self.ws,
            qkv=3, ws1=self.ws, ws2=self.ws)
        dots = (q @ k.transpose(-2, -1)) * self.scale
        if self.rpe:
            rpb = self.rpb_table[self.rpi.view(-1)].view(
                self.ws ** 2, self.ws ** 2, -1)
            dots += rpb.permute(2, 0, 1).unsqueeze(0)
        attn = (dots.softmax(-1) @ v)
        attn = rearrange(
            attn,
            '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)',
            h=self.num_heads, d=C // self.num_heads,
            hh=Hp // self.ws, ww=Wp // self.ws,
            ws1=self.ws, ws2=self.ws)
        attn = attn[:, :, :H, :W]
        out = (self.attn_x(F.pad(attn, (0, 0, 0, 1), 'reflect')) +
               self.attn_y(F.pad(attn, (0, 1, 0, 0), 'reflect')))
        out = self.proj(
            F.pad(out + local, (0, 1, 0, 1), 'reflect'))[:, :, :H, :W]
        return out


class Block(nn.Module):
    def __init__(self, dim=256, nh=16, mr=4., qkv_bias=False, drop=0.,
                 dp=0., act=nn.ReLU6, norm=nn.BatchNorm2d, ws=8):
        super().__init__()
        self.norm1 = norm(dim)
        self.attn = GlobalLocalAttention(dim, nh, qkv_bias, ws)
        self.drop_path = DropPath(dp) if dp > 0. else nn.Identity()
        self.mlp = Mlp(dim, int(dim * mr), dim, act, drop)
        self.norm2 = norm(dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class WF(nn.Module):
    def __init__(self, inc=128, dc=128, eps=1e-8):
        super().__init__()
        self.pre = Conv(inc, dc, 1)
        self.w = nn.Parameter(torch.ones(2))
        self.eps = eps
        self.post = ConvBNReLU(dc, dc, 3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        w = F.relu(self.w)
        w = w / (w.sum() + self.eps)
        return self.post(w[0] * self.pre(res) + w[1] * x)


class FeatureRefinementHead(nn.Module):
    def __init__(self, inc=64, dc=64):
        super().__init__()
        self.pre = Conv(inc, dc, 1)
        self.w = nn.Parameter(torch.ones(2))
        self.eps = 1e-8
        self.post = ConvBNReLU(dc, dc, 3)
        self.pa = nn.Sequential(
            nn.Conv2d(dc, dc, 3, padding=1, groups=dc), nn.Sigmoid())
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), Conv(dc, dc // 16, 1),
            nn.ReLU6(), Conv(dc // 16, dc, 1), nn.Sigmoid())
        self.sc = ConvBN(dc, dc, 1)
        self.proj = SeparableConvBN(dc, dc, 3)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        w = F.relu(self.w)
        w = w / (w.sum() + self.eps)
        x = self.post(w[0] * self.pre(res) + w[1] * x)
        return self.act(
            self.proj(self.pa(x) * x + self.ca(x) * x) + self.sc(x))


class Decoder(nn.Module):
    """
    Hierarchical global-local attention decoder with VLP prior fusion (Eq. 8):
        Y = Y_hat + sigmoid(gamma) * (sigmoid(w_sim) ⊙ S_sim^up
                                      + sigmoid(w_cls) ⊙ Y_dense^up).
    """

    def __init__(self, encoder_channels=(256, 256, 256, 256),
                 decode_channels=64, dropout=0.1, window_size=8, num_classes=6):
        super().__init__()
        self.num_classes = num_classes
        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, 1)
        self.b4 = Block(decode_channels, 8, ws=window_size)
        self.b3 = Block(decode_channels, 8, ws=window_size)
        self.p3 = WF(encoder_channels[-2], decode_channels)
        self.b2 = Block(decode_channels, 8, ws=window_size)
        self.p2 = WF(encoder_channels[-3], decode_channels)
        self.p1 = FeatureRefinementHead(encoder_channels[-4], decode_channels)
        self.seg = nn.Sequential(
            ConvBNReLU(decode_channels, decode_channels),
            nn.Dropout2d(dropout, True),
            Conv(decode_channels, num_classes, 1))

        # Eq. 8 fusion gates: w_sim, w_cls (per-class), gamma (overall)
        self.w_sim = nn.Parameter(torch.zeros(num_classes))
        self.w_cls = nn.Parameter(torch.ones(num_classes) * 0.5)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, r1, r2, r3, r4, h, w, vlp_output):
        # Y_hat: decoder output before VLP fusion
        x = self.b4(self.pre_conv(r4))
        x = self.b3(self.p3(x, r3))
        x = self.b2(self.p2(x, r2))
        y_hat = self.seg(self.p1(x, r1))

        # VLP priors -> upsample to Y_hat resolution
        sim_up = F.interpolate(vlp_output['sim_map'], y_hat.shape[-2:],
                               mode='bilinear', align_corners=False)
        dense_up = F.interpolate(vlp_output['dense_logits'], y_hat.shape[-2:],
                                 mode='bilinear', align_corners=False)

        # Eq. 8
        w_s = torch.sigmoid(self.w_sim).view(1, -1, 1, 1)
        w_c = torch.sigmoid(self.w_cls).view(1, -1, 1, 1)
        y = y_hat + torch.sigmoid(self.gamma) * (w_s * sim_up + w_c * dense_up)

        return F.interpolate(y, (h, w), mode='bilinear', align_corners=False)


# ============================================================
#  MAPLE main model  (registered as 'maple')
# ============================================================

@MODEL_REGISTRY.register('maple')
class MAPLE(BaseSegmentor):
    """
    MAPLE end-to-end framework (Sec. III-A overview):
        Frozen SAM3 + LMA(rgb,dsm) + LCAS + VLP + cross-modal fusion + Decoder
        + segmentation loss L_seg + auxiliary SAFR loss L_SAFR.

    Plugs into the framework via :class:`BaseSegmentor`:
        - ``forward_train(batch)`` consumes batch['image'], batch['dsm'],
          batch['text_prompts'], batch['label']; returns dict with
          {'loss', 'logits', 'aux_losses'}.
        - ``forward_test(batch)``  same inputs minus the label, returns logits.
        - ``param_groups()`` splits VLP (RemoteCLIP) parameters into a slow
          LR group, matching the paper's training protocol.
        - ``on_train_iter_end()`` forwards itself to a SAFRWeightLogger.
    """

    def __init__(self,
                 decode_channels=64, dropout=0.1, window_size=8, num_classes=6,
                 # SAM3 backbone
                 sam3_checkpoint_path=None, img_size=336, adapter_reduction=32,
                 # LCAS layer selection (default: align with SAM3 global-attn layers)
                 lcas_layers=None,
                 # VLP (RemoteCLIP)
                 clip_model_name="ViT-B-32", clip_checkpoint_path=None,
                 vlp_adapter_reduction=4,
                 dch_layers=1, dch_mlp_ratio=4.0,
                 # SAFR loss
                 safr_loss_weight=0.1, safr_warmup_steps=100,
                 safr_beta1=0.99, safr_beta2=0.9,
                 device='cuda'):
        super().__init__()
        assert img_size % 14 == 0

        self.img_size = img_size
        self.num_classes = num_classes

        # 1. Frozen SAM3 ViT backbone (theta_frozen)
        self.sam3_vit = create_sam3_vit(img_size)
        if sam3_checkpoint_path:
            load_sam3_weights(self.sam3_vit, sam3_checkpoint_path)
        for p in self.sam3_vit.parameters():
            p.requires_grad = False

        num_blocks = len(self.sam3_vit.blocks)
        dim = 1024

        # LCAS conditions only the layers in `lcas_layers`. Other layers keep
        # plain Houlsby-style LMA adapters. By default we align with SAM3's
        # own global-attention layers (7, 15, 23, 31), which are the natural
        # semantic-aggregation points in the backbone.
        if lcas_layers is None:
            lcas_layers = list(self.sam3_vit.full_attn_ids)
        self.lcas_layers = sorted(set(lcas_layers))

        # 2. Modality-specific LMA (Phi_rgb, Phi_dsm); one bottleneck per block
        self.lma_rgb = LMA(num_blocks, dim, adapter_reduction)
        self.lma_dsm = LMA(num_blocks, dim, adapter_reduction)

        # 3. VLP (RemoteCLIP) + LCAS condition synthesizer
        self.vlp = VLP(
            model_name=clip_model_name,
            checkpoint_path=clip_checkpoint_path,
            device=device,
            num_classes=num_classes,
            adapter_reduction=vlp_adapter_reduction,
            dch_layers=dch_layers,
            dch_mlp_ratio=dch_mlp_ratio)
        self.vlp_grid_size = self.vlp.grid_size

        # LCAS only stores per-class bases for the conditioned layers
        self.lcas = LCAS(
            num_classes=num_classes,
            num_lcas_layers=len(self.lcas_layers),
            adapter_bottleneck_dim=adapter_reduction,
            num_modalities=2)

        # 4. Reduce: 1024 -> 256 at four scales per modality
        self.reduce1_rgb = nn.Conv2d(1024, 256, 1)
        self.reduce2_rgb = nn.Conv2d(1024, 256, 1)
        self.reduce3_rgb = nn.Conv2d(1024, 256, 1)
        self.reduce4_rgb = nn.Conv2d(1024, 256, 1)
        self.reduce1_dsm = nn.Conv2d(1024, 256, 1)
        self.reduce2_dsm = nn.Conv2d(1024, 256, 1)
        self.reduce3_dsm = nn.Conv2d(1024, 256, 1)
        self.reduce4_dsm = nn.Conv2d(1024, 256, 1)

        # 5. Scale-wise cross-modal SE fusion
        self.fusion1 = SEFusion(256)
        self.fusion2 = SEFusion(256)
        self.fusion3 = SEFusion(256)
        self.fusion4 = SEFusion(256)

        # 6. Hierarchical decoder with VLP prior fusion (Eq. 8)
        self.decoder = Decoder(
            (256, 256, 256, 256), decode_channels, dropout,
            window_size, num_classes)

        # 7. SAFR auxiliary loss (Eq. 9-15)
        self.safr_loss = SAFRLoss(
            num_classes=num_classes,
            grid_size=self.vlp_grid_size,
            loss_weight=safr_loss_weight,
            warmup_steps=safr_warmup_steps,
            beta1=safr_beta1, beta2=safr_beta2)

    def forward(self, rgb, dsm, text_prompts, gt_mask=None):
        B, _, H, W = rgb.shape

        # 1. VLP forward -> S_sim, Y_dense
        vlp_output = self.vlp(rgb, text_prompts)
        sim_map = vlp_output['sim_map']

        # 2. LCAS: condition signals for both modalities
        film_attn_rgb, film_mlp_rgb, spatial_weights = \
            self.lcas(sim_map, modality_id=0)
        film_attn_dsm, film_mlp_dsm, _ = \
            self.lcas(sim_map, modality_id=1)

        # 3. DSM is single-channel: replicate to 3-channel pseudo-RGB
        if dsm.dim() == 3:
            dsm = dsm.unsqueeze(1)
        dsm_3ch = dsm.repeat(1, 3, 1, 1)

        # 4. Resize to backbone input size
        if H != self.img_size or W != self.img_size:
            rgb_in = F.interpolate(rgb, (self.img_size,) * 2,
                                   mode='bilinear', align_corners=False)
            dsm_in = F.interpolate(dsm_3ch, (self.img_size,) * 2,
                                   mode='bilinear', align_corners=False)
        else:
            rgb_in, dsm_in = rgb, dsm_3ch

        # 5. SAM3 forward with LMA injection (Eq. 3); LCAS conditions only
        #    the layers listed in self.lcas_layers, others use plain adapters.
        feat_rgb = forward_vit_with_lma(
            self.sam3_vit, rgb_in, self.lma_rgb, self.lcas_layers,
            film_attn_rgb, film_mlp_rgb, spatial_weights)[-1]
        feat_dsm = forward_vit_with_lma(
            self.sam3_vit, dsm_in, self.lma_dsm, self.lcas_layers,
            film_attn_dsm, film_mlp_dsm, spatial_weights)[-1]

        # 6. Multi-scale reduce + cross-modal fusion
        r1_rgb = F.interpolate(self.reduce1_rgb(feat_rgb), (H // 4,  W // 4),  mode='bilinear')
        r2_rgb = F.interpolate(self.reduce2_rgb(feat_rgb), (H // 8,  W // 8),  mode='bilinear')
        r3_rgb = F.interpolate(self.reduce3_rgb(feat_rgb), (H // 16, W // 16), mode='bilinear')
        r4_rgb = F.interpolate(self.reduce4_rgb(feat_rgb), (H // 32, W // 32), mode='bilinear')
        r1_dsm = F.interpolate(self.reduce1_dsm(feat_dsm), (H // 4,  W // 4),  mode='bilinear')
        r2_dsm = F.interpolate(self.reduce2_dsm(feat_dsm), (H // 8,  W // 8),  mode='bilinear')
        r3_dsm = F.interpolate(self.reduce3_dsm(feat_dsm), (H // 16, W // 16), mode='bilinear')
        r4_dsm = F.interpolate(self.reduce4_dsm(feat_dsm), (H // 32, W // 32), mode='bilinear')

        r1 = self.fusion1(r1_rgb, r1_dsm)
        r2 = self.fusion2(r2_rgb, r2_dsm)
        r3 = self.fusion3(r3_rgb, r3_dsm)
        r4 = self.fusion4(r4_rgb, r4_dsm)

        # 7. Decoder + VLP prior fusion (Eq. 8)
        output = self.decoder(r1, r2, r3, r4, H, W, vlp_output=vlp_output)

        # 8. Auxiliary SAFR loss during training (Eq. 14)
        if self.training and gt_mask is not None:
            l_safr = self.safr_loss(vlp_output, gt_mask)
            return output, l_safr

        return output

    # ------------------------------------------------------------------
    #  BaseSegmentor hooks
    # ------------------------------------------------------------------

    def _run(self, batch: Dict[str, Any], with_gt: bool):
        """Common entry shared by train/test paths.  ``batch`` is a dict
        produced by the dataset + populated by the trainer with
        ``batch['text_prompts']`` (list[str] of length num_classes)."""
        gt = batch['label'] if with_gt else None
        return self(batch['image'], batch['dsm'],
                    text_prompts=batch['text_prompts'], gt_mask=gt)

    def forward_train(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out = self._run(batch, with_gt=True)
        if isinstance(out, tuple):
            logits, aux_loss = out
        else:
            logits = out
            aux_loss = torch.zeros((), device=logits.device)
        seg_loss = self.seg_loss_fn(logits, batch['label'])
        total = seg_loss + aux_loss
        return {
            'loss': total,
            'logits': logits,
            'aux_losses': {
                'seg':  seg_loss.detach(),
                'safr': aux_loss.detach(),
            },
        }

    @torch.no_grad()
    def forward_test(self, batch: Dict[str, Any]) -> torch.Tensor:
        out = self._run(batch, with_gt=False)
        return out[0] if isinstance(out, tuple) else out

    def param_groups(self, base_lr: float, vlp_lr_scale: float = 0.1,
                     **_: Any) -> List[dict]:
        """VLP (RemoteCLIP) gets a smaller LR, as per the paper."""
        vlp_params, other_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (vlp_params if 'vlp' in name else other_params).append(p)
        return [
            {'params': other_params, 'lr': base_lr},
            {'params': vlp_params,   'lr': base_lr * vlp_lr_scale, 'name': 'vlp'},
        ]

    def on_train_iter_end(self, iteration: int,
                          logger: Optional[Any] = None) -> None:
        """Forward to a SAFRWeightLogger if one was supplied."""
        if logger is not None:
            logger.log(self, iteration)
