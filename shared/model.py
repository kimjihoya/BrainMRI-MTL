import os
import sys
import importlib.util

import torch
import torch.nn as nn
import torch.nn.functional as F

# -- BrainIAC ViT backbone --
_brainiac_path = os.path.join(os.path.dirname(__file__), '..', 'BrainIAC', 'src', 'model.py')
_spec = importlib.util.spec_from_file_location("brainiac_model", _brainiac_path)
_brainiac_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_brainiac_model)
ViTBackboneNet = _brainiac_model.ViTBackboneNet

# -- BrainMVP Uniformer backbone --
_brainmvp_dir = os.path.join(os.path.dirname(__file__), '..', 'BrainMVP')
if _brainmvp_dir not in sys.path:
    sys.path.insert(0, _brainmvp_dir)

# -- Triad PlainConvUNet backbone --
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Triad'))
_triad_path = os.path.join(os.path.dirname(__file__), '..', 'Triad', 'QuickStart.py')
_triad_spec = importlib.util.spec_from_file_location("triad_quickstart", _triad_path)
_triad_module = importlib.util.module_from_spec(_triad_spec)
_triad_spec.loader.exec_module(_triad_module)
get_Plain_nnUNet = _triad_module.get_Plain_nnUNet
PlainConvUNet   = _triad_module.PlainConvUNet

TRIAD_FEAT_DIM   = 32 + 64 + 128 + 256 + 320 + 320  # 1120
BRAINMVP_FEAT_DIM = 512

TASK_NAMES = ["end1_binary", "mortality_binary", "dementia_binary", "mace_binary"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_brainmvp(ckpt_path: str, num_phase: int = 1):
    from models.Uniformer import SSLEncoder
    encoder = SSLEncoder(num_phase=num_phase)
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]
        enc_sd = {k.replace("module.encoder.", ""): v
                  for k, v in sd.items() if k.startswith("module.encoder.")}
        missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
        print(f"BrainMVP loaded (missing={len(missing)}, unexpected={len(unexpected)}).")
    return encoder


def _brainmvp_encode(backbone, x: torch.Tensor) -> torch.Tensor:
    _, _, _, _, x_enc4 = backbone(x)
    b = x_enc4.size(0)
    return F.adaptive_avg_pool3d(x_enc4, (1, 1, 1)).view(b, -1)  # (B, 512)


def _load_triad(ckpt_path: str) -> PlainConvUNet:
    model = get_Plain_nnUNet(num_input_channels=1)
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True)
    print("Triad PlainConvUNet weights loaded.")
    return model


def _triad_encode(backbone: PlainConvUNet, x: torch.Tensor,
                  pool: str = "avg") -> torch.Tensor:
    # avg pooling dilutes small acute lesions into the whole-brain mean (linear-probe diagnostic:
    # END1 DWI 0.478 -> 0.579 with max pooling). pool="max" preserves the local peak.
    skips = backbone.encoder(x)
    b = skips[0].size(0)
    pool_fn = F.adaptive_max_pool3d if pool == "max" else F.adaptive_avg_pool3d
    return torch.cat(
        [pool_fn(s, (1, 1, 1)).view(b, -1) for s in skips], dim=1
    )  # (B, 1120)


class TabularEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 output_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_proj(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.GELU(),
        nn.Dropout(dropout),
    )


class TaskAdapter(nn.Module):
    """Task-specific bottleneck residual adapter (inserted between fusion and head)."""
    def __init__(self, feat_dim: int, bottleneck: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, feat_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class CrossModalAttention(nn.Module):
    """Clinical (tab) feature attends to the image/DWI context."""
    def __init__(self, tab_dim: int, context_dims: list,
                 attn_dim: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        attn_dim = (attn_dim // n_heads) * n_heads  # divisibility guarantee
        self.q_proj  = nn.Linear(tab_dim, attn_dim)
        self.kv_projs = nn.ModuleList([nn.Linear(d, attn_dim) for d in context_dims])
        self.attn    = nn.MultiheadAttention(attn_dim, n_heads, dropout=dropout, batch_first=True)
        self.out_proj = nn.Linear(attn_dim, tab_dim)
        self.norm    = nn.LayerNorm(tab_dim)

    def forward(self, tab: torch.Tensor, context_feats: list) -> torch.Tensor:
        q   = self.q_proj(tab).unsqueeze(1)                                   # [B, 1, attn_dim]
        kv  = torch.cat([p(f).unsqueeze(1) for p, f
                          in zip(self.kv_projs, context_feats)], dim=1)       # [B, n_ctx, attn_dim]
        out, _ = self.attn(q, kv, kv)                                         # [B, 1, attn_dim]
        out = self.out_proj(out.squeeze(1))                                    # [B, tab_dim]
        return self.norm(tab + out)


class FiLMLayer(nn.Module):
    """Clinical variables (condition) modulate the image feature element-wise (FiLM)."""
    def __init__(self, condition_dim: int, feature_dim: int):
        super().__init__()
        self.gamma = nn.Linear(condition_dim, feature_dim)
        self.beta  = nn.Linear(condition_dim, feature_dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.gamma(condition) * x + self.beta(condition)


class TaskConditionedFusion(nn.Module):
    """
    Task-conditioned Modality MoE.

    Each task dynamically weights the [T2, DWI, Tab] branches based on the input.
      1. shared proj: each modality -> fusion_dim
      2. per-task router: concat(projs) -> softmax -> (B, n_branches) attention
      3. weighted sum → post-norm
    """
    def __init__(self, branch_dims: list, fusion_dim: int,
                 task_names: list, dropout: float,
                 task_specific_proj: bool = False):
        super().__init__()
        n = len(branch_dims)
        self.n_branches = n
        self.task_specific = task_specific_proj

        if task_specific_proj:
            # per-task independent modality projection — removes shared-proj contamination
            self.modal_projs = nn.ModuleDict({
                name: nn.ModuleList([_make_proj(d, fusion_dim, dropout) for d in branch_dims])
                for name in task_names
            })
        else:
            self.modal_projs = nn.ModuleList([
                _make_proj(d, fusion_dim, dropout) for d in branch_dims
            ])
        # per task: a light linear router (input-dependent)
        self.routers = nn.ModuleDict({
            name: nn.Linear(fusion_dim * n, n, bias=True)
            for name in task_names
        })
        self.post_norm = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, parts: list, task_name: str, return_gate: bool = False):
        mp = self.modal_projs[task_name] if self.task_specific else self.modal_projs
        proj = [mp[i](parts[i]) for i in range(self.n_branches)]
        gates = torch.softmax(
            self.routers[task_name](torch.cat(proj, dim=1)), dim=1
        )  # (B, n_branches)
        fused = sum(gates[:, i:i+1] * proj[i] for i in range(self.n_branches))
        out = self.post_norm(fused)
        if return_gate:
            return out, gates
        return out


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class MultiTaskModel(nn.Module):
    """
    Configurable multi-branch Late-Fusion model.

    t2_backbone : "brainiac" | "triad" | "none"
    dwi_backbone: "triad"    | "none"

    Possible combinations:
        A. brainiac / none  — T2(BrainIAC) + clinical
        B. triad    / none  — T2(Triad)    + clinical
        C. none     / triad — DWI(Triad)   + clinical
        D. brainiac / triad — T2(BrainIAC) + DWI(Triad) + clinical
        E. triad    / triad — T2(Triad)    + DWI(Triad) + clinical
    """

    def __init__(
        self,
        t2_backbone:      str  = "brainiac",
        dwi_backbone:     str  = "triad",
        ckpt_path:        str  = None,   # BrainIAC
        triad_ckpt_path:  str  = None,   # Triad
        brainmvp_ckpt_path: str = None,  # BrainMVP
        task_names:     list = TASK_NAMES,
        tab_input_dim:  int  = 39,
        image_feat_dim: int  = 256,
        dwi_feat_dim:   int  = 256,
        tab_feat_dim:   int  = 256,
        fusion_dim:     int  = 512,
        dropout:        float = 0.3,
        fusion:             str  = "gated",
        task_specific_proj: bool = False,  # moe: per-task independent modality projection
        task_dwi_residual:  bool = False,
        deep_head_tasks:    list = None,
        film:               bool = False,  # condition image features on clinical variables
        key_feature_indices: list = None,  # indices of key features to skip-connect
        key_feature_tasks:   list = None,  # tasks to which the skip connection applies
        task_modality_gate:  bool = False,  # learn per-task modality weights
        task_adapters:       bool = False,  # per-task bottleneck residual adapter
        cross_attn:          bool = False,  # tab attends to the image/DWI context
        cross_attn_dim:      int  = 128,    # internal cross-attention dim
        triad_pool:          str  = "avg",  # Triad encoder pooling: "avg" | "max"
        ordinal_tasks:       list = None,   # #9: CORN ordinal heads (K-1 logits) for these tasks
        ordinal_num_classes: int  = 7,      # mRS 0-6 → 7 classes → 6 logits
    ):
        super().__init__()
        self.task_names   = task_names
        self.t2_backbone  = t2_backbone.lower()
        self.dwi_backbone = dwi_backbone.lower()
        self.triad_pool   = triad_pool.lower()

        assert self.t2_backbone in ("brainiac", "triad", "brainmvp", "none")
        assert self.dwi_backbone in ("triad", "none")
        assert self.t2_backbone != "none" or self.dwi_backbone != "none", \
            "at least one of t2_backbone / dwi_backbone must be set"

        # ── T2 branch ──────────────────────────────────────────────────────
        if self.t2_backbone == "brainiac":
            self.backbone   = ViTBackboneNet(simclr_ckpt_path=ckpt_path)
            self.image_proj = _make_proj(768, image_feat_dim, dropout)
        elif self.t2_backbone == "triad":
            self.backbone   = _load_triad(triad_ckpt_path)
            self.image_proj = _make_proj(TRIAD_FEAT_DIM, image_feat_dim, dropout)
        elif self.t2_backbone == "brainmvp":
            self.backbone   = _load_brainmvp(brainmvp_ckpt_path, num_phase=1)
            self.image_proj = _make_proj(BRAINMVP_FEAT_DIM, image_feat_dim, dropout)

        # ── DWI branch ─────────────────────────────────────────────────────
        if self.dwi_backbone == "triad":
            self.dwi_enc  = _load_triad(triad_ckpt_path)
            self.dwi_proj = _make_proj(TRIAD_FEAT_DIM, dwi_feat_dim, dropout)
            # #7: learned missing-DWI embedding. Replaces the encoder output for
            # patients without DWI (currently zeros → conv bias → "ghost feature").
            # Index of has_dwi within the tab vector = |cont| + |bin| (see ALL_FEATURES).
            from dataset import CONT_FEATURES as _CF, BIN_FEATURES as _BF
            self._has_dwi_idx = len(_CF) + len(_BF)
            self.missing_dwi_emb = nn.Parameter(torch.zeros(dwi_feat_dim))

        # ── Tabular branch ─────────────────────────────────────────────────
        self.tab_encoder = TabularEncoder(
            input_dim=tab_input_dim,
            hidden_dim=max(64, tab_input_dim * 2),
            output_dim=tab_feat_dim,
            dropout=dropout,
        )

        # ── FiLM ───────────────────────────────────────────────────────────
        if film:
            if self.t2_backbone != "none":
                self.film_t2  = FiLMLayer(tab_feat_dim, image_feat_dim)
            if self.dwi_backbone != "none":
                self.film_dwi = FiLMLayer(tab_feat_dim, dwi_feat_dim)

        # ── Cross-Modal Attention ───────────────────────────────────────────
        if cross_attn:
            ctx_dims = []
            if self.t2_backbone  != "none": ctx_dims.append(image_feat_dim)
            if self.dwi_backbone != "none": ctx_dims.append(dwi_feat_dim)
            if ctx_dims:
                n_heads = max(1, cross_attn_dim // 32)
                self.cross_modal_attn = CrossModalAttention(
                    tab_dim=tab_feat_dim, context_dims=ctx_dims,
                    attn_dim=cross_attn_dim, n_heads=n_heads,
                )

        # ── Fusion ─────────────────────────────────────────────────────────
        self.fusion_type = fusion.lower()
        assert self.fusion_type in ("mlp", "gated", "moe", "ensemble")

        # #9: CORN ordinal heads. Ordinal tasks emit (num_classes-1) logits instead
        # of a single regression scalar; everything else stays 1-d.
        self.ordinal_tasks = set(ordinal_tasks or [])
        self.ordinal_K = int(ordinal_num_classes)
        def _out_dim(name):
            return (self.ordinal_K - 1) if name in self.ordinal_tasks else 1
        self._out_dim = _out_dim

        # ensemble fusion returns per-branch soft-voted logits and never reaches the
        # fused-vector path, so these options would be silently no-ops. Fail loudly
        # instead of pretending they took effect (code-review code-quality item).
        if self.fusion_type == "ensemble":
            _ignored = [n for n, v in (("key_feature_indices", key_feature_indices),
                                       ("task_dwi_residual", task_dwi_residual),
                                       ("task_modality_gate", task_modality_gate)) if v]
            assert not _ignored, (
                f"fusion='ensemble' ignores {_ignored}; remove them or pick mlp/gated/moe.")

        # branch dims in order: [t2, dwi, tab] (only active branches)
        branch_dims = []
        if self.t2_backbone  != "none": branch_dims.append(image_feat_dim)
        if self.dwi_backbone != "none": branch_dims.append(dwi_feat_dim)
        branch_dims.append(tab_feat_dim)
        self._n_branches = len(branch_dims)

        if self.fusion_type == "mlp":
            feat_dim = sum(branch_dims)
            self.fusion_mlp = nn.Sequential(
                nn.Linear(feat_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        elif self.fusion_type == "gated":
            # project every branch to the same fusion_dim
            proj_list = [_make_proj(d, fusion_dim, dropout) for d in branch_dims]
            self.gate_projs = nn.ModuleList(proj_list)
            # gate network: concat of projected features → n_branches scalar gates
            self.gate_net = nn.Sequential(
                nn.Linear(fusion_dim * self._n_branches, self._n_branches),
            )
            # lightweight post-fusion MLP
            self.fusion_mlp = nn.Sequential(
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        elif self.fusion_type == "moe":
            self.task_fusion = TaskConditionedFusion(
                branch_dims=branch_dims,
                fusion_dim=fusion_dim,
                task_names=task_names,
                dropout=dropout,
                task_specific_proj=task_specific_proj,
            )

        else:  # ensemble
            # each branch predicts all tasks independently, then weighted soft voting
            self.branch_projs = nn.ModuleList([
                _make_proj(d, fusion_dim, dropout) for d in branch_dims
            ])
            self.branch_heads = nn.ModuleList([
                nn.ModuleDict({name: nn.Linear(fusion_dim, _out_dim(name)) for name in task_names})
                for _ in branch_dims
            ])
            # (n_branches, n_tasks) — softmax over dim=0 -> per-task branch contribution
            self.ensemble_weights = nn.Parameter(
                torch.zeros(self._n_branches, len(task_names))
            )

        # ── Task-Specific Adapters ──────────────────────────────────────────
        if task_adapters:
            self.task_adapters = nn.ModuleDict({
                name: TaskAdapter(fusion_dim, bottleneck=64, dropout=dropout)
                for name in task_names
            })

        if self.fusion_type != "ensemble":
            deep = set(deep_head_tasks or [])
            self.heads = nn.ModuleDict({
                name: (
                    nn.Sequential(
                        nn.Linear(fusion_dim, fusion_dim // 2),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(fusion_dim // 2, _out_dim(name)),
                    ) if name in deep else nn.Linear(fusion_dim, _out_dim(name))
                )
                for name in task_names
            })

        # ── Key feature skip connection ────────────────────────────────────
        # wire key clinical variables (e.g. ini_nih, pre_mrs) directly into a task head
        # -> force that task to always reference these variables
        if key_feature_indices:
            self.key_feat_idx   = key_feature_indices
            self.key_feat_tasks = set(key_feature_tasks or task_names)
            self.key_feat_proj  = nn.Linear(len(key_feature_indices), fusion_dim, bias=False)

        # ── Task-specific modality gate ───────────────────────────────────
        # each task independently learns how much to reference the [t2, dwi, tab] branches
        # for tasks where tabular matters more than MRI (e.g. MACE), tab weight rises automatically
        if task_modality_gate:
            self.modal_projs = nn.ModuleList([
                _make_proj(d, fusion_dim, dropout) for d in branch_dims
            ])
            self.task_modal_gates = nn.ParameterDict({
                name: nn.Parameter(torch.zeros(self._n_branches))
                for name in task_names
            })
            self.task_gate_norm = nn.Sequential(
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        # ── Task-specific DWI residual ─────────────────────────────────────
        # each head additionally references the DWI feature on top of the shared fusion
        # tasks with high DWI dependence (e.g. END1) can learn a larger residual
        if self.dwi_backbone != "none" and task_dwi_residual:
            self.task_dwi_res = nn.ModuleDict({
                name: nn.Linear(dwi_feat_dim, fusion_dim, bias=False)
                for name in task_names
            })

        mode_str = f"t2={t2_backbone} / dwi={dwi_backbone} / fusion={fusion}"
        print(f"MultiTaskModel init: {mode_str}  |  branches={self._n_branches}")

    def forward(self, x: torch.Tensor, dwi: torch.Tensor,
                tab: torch.Tensor, return_gates: bool = False) -> dict:
        parts = []
        t2_feat = None
        gate_store = {}

        # encode tab first (used as the FiLM condition)
        tab_feat = self.tab_encoder(tab)

        if self.t2_backbone == "brainiac":
            t2_feat = self.image_proj(self.backbone(x))
        elif self.t2_backbone == "triad":
            t2_feat = self.image_proj(_triad_encode(self.backbone, x, self.triad_pool))
        elif self.t2_backbone == "brainmvp":
            t2_feat = self.image_proj(_brainmvp_encode(self.backbone, x))

        if self.t2_backbone != "none":
            if hasattr(self, "film_t2"):
                t2_feat = self.film_t2(t2_feat, tab_feat)
            parts.append(t2_feat)

        _dwi_feat = None
        if self.dwi_backbone == "triad":
            _dwi_feat = self.dwi_proj(_triad_encode(self.dwi_enc, dwi, self.triad_pool))
            # #7: swap in the learned missing embedding for has_dwi==0 rows so the
            # zeros-volume "ghost feature" never reaches fusion.
            if hasattr(self, "missing_dwi_emb"):
                has_dwi = tab[:, self._has_dwi_idx:self._has_dwi_idx + 1]  # (B,1)
                _dwi_feat = has_dwi * _dwi_feat + (1.0 - has_dwi) * self.missing_dwi_emb.unsqueeze(0)
            if hasattr(self, "film_dwi"):
                _dwi_feat = self.film_dwi(_dwi_feat, tab_feat)
            parts.append(_dwi_feat)

        # Cross-modal attention: tab is enriched by attending to the image/DWI context
        if hasattr(self, 'cross_modal_attn'):
            ctx = [f for f in [t2_feat, _dwi_feat] if f is not None]
            if ctx:
                tab_feat = self.cross_modal_attn(tab_feat, ctx)

        parts.append(tab_feat)

        # -- ensemble: each branch predicts independently -> weighted soft voting --
        if self.fusion_type == "ensemble":
            branch_feats = [self.branch_projs[i](parts[i]) for i in range(self._n_branches)]
            weights = torch.softmax(self.ensemble_weights, dim=0)  # (n_branches, n_tasks)
            if hasattr(self, 'task_adapters'):
                return {
                    name: sum(
                        weights[i, j] * self.branch_heads[i][name](
                            self.task_adapters[name](branch_feats[i])
                        )
                        for i in range(self._n_branches)
                    )
                    for j, name in enumerate(self.task_names)
                }
            return {
                name: sum(
                    weights[i, j] * self.branch_heads[i][name](branch_feats[i])
                    for i in range(self._n_branches)
                )
                for j, name in enumerate(self.task_names)
            }

        if self.fusion_type == "mlp":
            fused = self.fusion_mlp(torch.cat(parts, dim=1))
        elif self.fusion_type == "gated":
            proj = [self.gate_projs[i](parts[i]) for i in range(self._n_branches)]
            gates = torch.sigmoid(self.gate_net(torch.cat(proj, dim=1)))
            fused = sum(gates[:, i:i+1] * proj[i] for i in range(self._n_branches))
            fused = self.fusion_mlp(fused)
        else:  # moe: per-task routing — fused computed inside _head
            fused = None

        # key feature skip: wire key clinical variables directly into the task head
        key_skip = None
        if hasattr(self, "key_feat_proj"):
            key_skip = self.key_feat_proj(tab[:, self.key_feat_idx])

        # task-specific modality gate (legacy additive, for non-moe mode)
        modal_proj = None
        if hasattr(self, "modal_projs") and self.fusion_type != "moe":
            modal_proj = [self.modal_projs[i](parts[i]) for i in range(self._n_branches)]

        def _head(name: str) -> torch.Tensor:
            if self.fusion_type == "moe":
                if return_gates:
                    h, g = self.task_fusion(parts, name, return_gate=True)
                    gate_store[name] = g
                else:
                    h = self.task_fusion(parts, name)
            else:
                h = fused
            if hasattr(self, 'task_adapters'):
                h = self.task_adapters[name](h)
            if hasattr(self, "task_dwi_res") and _dwi_feat is not None:
                h = h + self.task_dwi_res[name](_dwi_feat)
            if key_skip is not None and name in self.key_feat_tasks:
                h = h + key_skip
            if modal_proj is not None:
                static_gates = torch.softmax(self.task_modal_gates[name], dim=0)
                task_fused = sum(static_gates[i] * modal_proj[i] for i in range(self._n_branches))
                h = h + self.task_gate_norm(task_fused)
            return self.heads[name](h)

        outputs = {name: _head(name) for name in self.task_names}
        if return_gates:
            return outputs, gate_store
        return outputs
