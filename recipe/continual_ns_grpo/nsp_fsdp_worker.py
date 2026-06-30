import asyncio
import contextlib
import logging
import os
from collections import defaultdict
import functools
import re
from typing import Optional

from contextlib import contextmanager
from recipe.continual_ns_grpo.nsp_config import NullSpaceProjConfig
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict
import torch.distributed
from dataclasses import fields as dataclass_fields
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._unshard_param_utils import _get_module_fsdp_state, _unshard_params_for_summon
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_id, get_device_name, get_torch_device, set_expandable_segments
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fsdp_utils import fsdp_version
from verl.utils.import_utils import import_external_libs
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.profiler import DistProfiler, log_gpu_memory_usage, simple_timer
from verl.utils.profiler.performance import reduce_timing, topk_reduce_ratio_min_max
from verl.workers.config import HFModelConfig
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
from verl.utils.model import compute_position_id_with_mask
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    collect_lora_params,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    get_shard_placement_fn,
    init_fn,
    layered_summon_lora_params,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
    replace_lora_wrapper,
)
from recipe.continual_ns_grpo.utils import QwenModuleEligibilityChecker


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


class NSPActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    def __init__(self, config, role: str, **kwargs):
        super().__init__(config, role, **kwargs)

        # NSP is only meaningful on actor workers
        # The config here is actor_rollout_ref config, not the full config
        raw_nsp_cfg = None
        if hasattr(config, "nsp"):
            raw_nsp_cfg = getattr(config, "nsp", None)

        print(f"[NSP Worker] Init on {device_name}, rank={self.rank}/{self.world_size}")
        
        # Debug: Check dispatch info
        if hasattr(self, '_dispatch_info_dict'):
            print(f"[NSP Worker] Rank {self.rank} dispatch_info: {self._dispatch_info_dict}")
        else:
            print(f"[NSP Worker] Rank {self.rank} WARNING: No _dispatch_info_dict found!")

        if raw_nsp_cfg is not None:
            cfg_dict = OmegaConf.to_container(raw_nsp_cfg, resolve=True)
            allowed_keys = {f.name for f in dataclass_fields(NullSpaceProjConfig)}
            filtered = {k: v for k, v in cfg_dict.items() if k in allowed_keys}
            self.nsp_cfg = NullSpaceProjConfig(**filtered)
        else:
            self.nsp_cfg = NullSpaceProjConfig()
        self.enable_nullspace_projection = bool(self.nsp_cfg.enable_nullspace_projection)
        self.token_importance = bool(getattr(self.nsp_cfg, "token_importance", False))
        self.token_importance_top_ratio = float(getattr(self.nsp_cfg, "token_importance_top_ratio", 1.0) or 1.0)
        self.layer_adaptivity = bool(getattr(self.nsp_cfg, "layer_adaptivity", False))
        self.layer_selection_metric = str(getattr(self.nsp_cfg, "layer_selection_metric", "mean") or "mean").lower()
        self.layer_top_k = int(getattr(self.nsp_cfg, "layer_top_k", 0) or 0)
        self.layer_top_ratio = float(getattr(self.nsp_cfg, "layer_top_ratio", 0.0) or 0.0)
        self.feature_level_adaptivity = bool(getattr(self.nsp_cfg, "block_null_space_projection", True))
        self.feature_activity_metric = str(getattr(self.nsp_cfg, "feature_activity_metric", "variance") or "variance").lower()
        self.high_freq_block_ratio = float(getattr(self.nsp_cfg, "high_freq_block_ratio", 1.0) or 1.0)
        self.high_freq_min_blocks = int(getattr(self.nsp_cfg, "high_freq_min_blocks", 1) or 1)
        self.nsp_log_level = str(getattr(self.nsp_cfg, "nsp_log_level", "basic") or "basic").lower()
        self._token_select_debug_counter = 0

        # Per-module covariance statistics (kept on actor rank)
        self.fea_in = defaultdict(dict)
        self.total_num_for_cov_update = {}
        self.activation_stats = defaultdict(dict)
        self.allowed_layer_ids: Optional[set[int]] = None
        self.selected_feature_indices: dict[str, torch.Tensor] = {}
        self._skip_next_nsp_reset_after_resume = False
        self._layer_pattern = re.compile(r"model\.layers\.(\d+)\.")

        # Forward-hook handles
        self.handles = []

        # Initialize module eligibility checker for NSP
        if self._is_actor and self.enable_nullspace_projection:
            # Module eligibility checker (which layers/modules participate in NSP)
            self.eligible_checker = QwenModuleEligibilityChecker(
                start_MLP=(
                    self.nsp_cfg.NSP_mlp_layer_range[0]
                    if self.nsp_cfg.NSP_mlp_layer_range
                    else None
                ),
                end_MLP=(
                    self.nsp_cfg.NSP_mlp_layer_range[1]
                    if self.nsp_cfg.NSP_mlp_layer_range
                    else None
                ),
                start_attention=(
                    self.nsp_cfg.NSP_attention_layer_range[0]
                    if self.nsp_cfg.NSP_attention_layer_range
                    else None
                ),
                end_attention=(
                    self.nsp_cfg.NSP_attention_layer_range[1]
                    if self.nsp_cfg.NSP_attention_layer_range
                    else None
                ),
            )
        else:
            self.eligible_checker = None

        self._log(
            "worker_init: enabled=%s, token_importance=%s, token_importance_top_ratio=%.3f, layer_adaptivity=%s, layer_selection_metric=%s, feature_level_adaptivity=%s, feature_activity_metric=%s, high_freq_block_ratio=%.3f, high_freq_min_blocks=%d, nsp_log_level=%s",
            self.enable_nullspace_projection,
            self.token_importance,
            float(self.token_importance_top_ratio),
            self.layer_adaptivity,
            self.layer_selection_metric,
            self.feature_level_adaptivity,
            self.feature_activity_metric,
            float(self.high_freq_block_ratio),
            int(self.high_freq_min_blocks),
            self.nsp_log_level,
        )

    def _log(self, msg: str, *args) -> None:
        if getattr(self, "rank", 0) != 0:
            return
        text = msg % args if args else msg
        print(f"[NSP Worker] {text}")

    def _log_detail(self, msg: str, *args) -> None:
        if self.nsp_log_level not in ("detail", "trace"):
            return
        self._log(msg, *args)

    def _log_trace(self, msg: str, *args) -> None:
        if self.nsp_log_level != "trace":
            return
        self._log(msg, *args)

    def check_whether_module_name_is_eligible_for_null_space_projection(self, module_name, module):
        if not self.enable_nullspace_projection:
            return False
        if self.eligible_checker is None:
            return False
        if not self.eligible_checker.check(module_name, module):
            return False
        if self.allowed_layer_ids is None:
            return True
        layer_idx = self._extract_layer_index(module_name)
        if layer_idx is None:
            return False
        return layer_idx in self.allowed_layer_ids

    @property
    def nsp_model(self):
        """Return the actor FSDP module used for NSP hooks."""
        return getattr(self, "actor_module_fsdp", None)

    @property
    def to_be_updated_modules(self):
        model = self.nsp_model
        if model is None or not self.enable_nullspace_projection:
            return {}
        return {
            n: m
            for n, m in model.named_modules()
            if self.check_whether_module_name_is_eligible_for_null_space_projection(n, m)
        }

    @property
    def nsp_stat_modules(self):
        model = self.nsp_model
        if model is None or not self.enable_nullspace_projection:
            return {}
        return {
            n: m
            for n, m in model.named_modules()
            if self.eligible_checker is not None and self.eligible_checker.check(n, m)
        }

    def update_cov(self, feat: torch.Tensor, name: str, weight_sum: float | None = None) -> torch.Tensor:
        """Accumulate a single covariance matrix per eligible module."""
        if not self.enable_nullspace_projection:
            return feat

        feat = feat.detach()
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)
        if feat.dim() != 2 or feat.numel() == 0:
            return feat

        if name not in self.fea_in:
            self.fea_in[name] = torch.zeros(
                (feat.shape[1], feat.shape[1]),
                device=feat.device,
                dtype=feat.dtype,
            )

        if name not in self.total_num_for_cov_update:
            self.total_num_for_cov_update[name] = 0.0

        self.fea_in[name] += torch.mm(feat.T, feat)
        denom_total = float(weight_sum) if weight_sum is not None else float(feat.shape[0])
        self.total_num_for_cov_update[name] += max(1.0, denom_total)
        return feat

    def _hard_select_tokens(self, feat: torch.Tensor, name: str) -> torch.Tensor:
        # Token-level 选择:
        # 对同一层收集到的 token 表征按 L2 norm 排序，只保留 top-ratio 的 token
        # 进入后续协方差估计。直觉上，这等价于优先用“激活更强”的 token
        # 去定义 anchor 子空间，弱 token 对 NSP 保护方向的影响被降低。
        if not self.token_importance:
            return feat

        feat = feat.detach()
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)
        if feat.dim() != 2 or feat.numel() == 0:
            return feat
        if feat.size(0) <= 1:
            self._log_trace("token_importance skipped: module=%s, token_count=%d", name, int(feat.size(0)))
            return feat

        keep_ratio = float(min(max(self.token_importance_top_ratio, 0.0), 1.0))
        keep_n = max(1, int(feat.size(0) * keep_ratio))
        keep_n = min(keep_n, feat.size(0))
        if keep_n >= feat.size(0):
            self._log_detail(
                "token_importance passthrough: module=%s, keep_n=%d/%d, ratio=%.3f",
                name,
                int(keep_n),
                int(feat.size(0)),
                keep_ratio,
            )
            return feat

        norms = torch.norm(feat, p=2, dim=1)
        top_vals, top_idx = torch.topk(norms, keep_n, largest=True, sorted=False)
        selected = feat.index_select(0, top_idx)
        # 这里是 README 中 token-level adaptivity 的简化实现版本：
        # 没有显式做 soft weighting，而是采用 hard top-k 保留关键 token。
        self._token_select_debug_counter += 1
        self._log_detail(
            "token_importance select: module=%s, keep_n=%d/%d, ratio=%.3f, norm_mean=%.6f, selected_norm_mean=%.6f",
            name,
            int(keep_n),
            int(feat.size(0)),
            keep_ratio,
            float(norms.mean().item()),
            float(top_vals.mean().item()),
        )
        self._log_trace(
            "token_importance trace: module=%s, norm_min=%.6f, norm_max=%.6f, selected_norm_min=%.6f, selected_norm_max=%.6f, event=%d",
            name,
            float(norms.min().item()),
            float(norms.max().item()),
            float(top_vals.min().item()),
            float(top_vals.max().item()),
            int(self._token_select_debug_counter),
        )
        return selected

    def _extract_layer_index(self, module_name: str) -> Optional[int]:
        match = self._layer_pattern.search(module_name or "")
        if not match:
            return None
        try:
            return int(match.group(1))
        except Exception:
            return None

    def _update_activation_stats(self, name: str, feat: torch.Tensor, weight_sum: float | None = None) -> None:
        if not self.enable_nullspace_projection:
            return

        feat = feat.detach()
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)
        if feat.dim() != 2 or feat.numel() == 0:
            return

        count = float(weight_sum) if weight_sum is not None else float(feat.shape[0])
        count = max(1.0, count)
        # Layer-level / feature-level 选择都不直接看完整协方差，而是先维护更轻量的
        # 一阶/二阶统计量:
        #   sum(feat), sum(feat^2), count
        # 这样后面就能恢复 mean 和 variance，用来给层、维度分别打分。
        sum_feat = feat.sum(dim=0).float()
        sumsq_feat = feat.float().pow(2).sum(dim=0)
        stats = self.activation_stats.get(name)
        if not stats:
            self.activation_stats[name] = {
                "sum": sum_feat,
                "sumsq": sumsq_feat,
                "count": count,
            }
            return

        if stats["sum"].device != sum_feat.device:
            stats["sum"] = stats["sum"].to(sum_feat.device)
        if stats["sumsq"].device != sumsq_feat.device:
            stats["sumsq"] = stats["sumsq"].to(sumsq_feat.device)
        stats["sum"] = stats["sum"] + sum_feat
        stats["sumsq"] = stats["sumsq"] + sumsq_feat
        stats["count"] = float(stats.get("count", 0.0)) + count

    def _finalize_layer_selection(
        self,
        layer_scores: dict[int, float],
        num_samples: int,
        tag: str,
        layer_module_counts: Optional[dict[int, int]] = None,
    ) -> dict:
        # Layer-level 选择:
        # 先把同一 transformer layer 下多个 eligible module 的分数汇总，
        # 再按 top-k 或 top-ratio 选出真正启用 NSP 的层集合。
        # 未入选的层在 projection 阶段保持原始梯度，不做额外保护。
        if not self.layer_adaptivity:
            self.allowed_layer_ids = None
            self._log("layer_selection skipped: layer_adaptivity disabled")
            return {"selected_layers": 0, "score_metric": tag, "selection_mode": "all"}

        if num_samples == 0 or not layer_scores:
            self.allowed_layer_ids = None
            self._log("layer_selection skipped: no %s scores collected; keep all layers", tag)
            return {"selected_layers": 0, "score_metric": tag, "selection_mode": "all"}

        for layer_id in list(layer_scores.keys()):
            layer_scores[layer_id] = layer_scores[layer_id] / float(max(1, num_samples))

        sorted_layers = sorted(layer_scores.items(), key=lambda kv: kv[1], reverse=True)
        if self.layer_top_k > 0:
            selected = [layer_id for layer_id, _ in sorted_layers[: self.layer_top_k]]
            selection_mode = f"top_k={self.layer_top_k}"
        elif self.layer_top_ratio > 0:
            keep_n = max(1, int(len(sorted_layers) * self.layer_top_ratio))
            selected = [layer_id for layer_id, _ in sorted_layers[:keep_n]]
            selection_mode = f"top_ratio={self.layer_top_ratio:.3f}"
        else:
            selected = [layer_id for layer_id, _ in sorted_layers]
            selection_mode = "all"

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            obj = [selected]
            torch.distributed.broadcast_object_list(obj, src=0)
            selected = obj[0]

        self.allowed_layer_ids = set(int(x) for x in selected)
        dropped = [int(layer_id) for layer_id, _ in sorted_layers if int(layer_id) not in self.allowed_layer_ids]
        selected_sorted = sorted(int(x) for x in self.allowed_layer_ids)
        cutoff_score = None
        if selected_sorted:
            for layer_id, score in sorted_layers:
                if int(layer_id) in self.allowed_layer_ids:
                    cutoff_score = float(score)
            cutoff_score = float(cutoff_score) if cutoff_score is not None else None
        preview = sorted_layers[: min(10, len(sorted_layers))]
        self._log(
            "layer_selection done: metric=%s, selected_layers=%d/%d, mode=%s, cutoff_score=%s",
            tag,
            len(self.allowed_layer_ids),
            len(sorted_layers),
            selection_mode,
            "n/a" if cutoff_score is None else f"{cutoff_score:.6f}",
        )
        self._log_detail(
            "layer_selection preview: %s",
            [(int(layer_id), float(score)) for layer_id, score in preview],
        )
        self._log_detail("layer_selection selected_layer_ids: %s", selected_sorted)
        self._log_detail("layer_selection dropped_layer_ids: %s", dropped)
        if layer_module_counts:
            selected_module_counts = {
                int(layer_id): int(layer_module_counts.get(int(layer_id), 0))
                for layer_id in selected_sorted
            }
            self._log_detail("layer_selection selected_layer_module_counts: %s", selected_module_counts)
        self._log_trace(
            "layer_selection full_scores: %s",
            [(int(layer_id), float(score)) for layer_id, score in sorted_layers],
        )
        return {
            "selected_layers": len(self.allowed_layer_ids),
            "total_layers_scored": len(sorted_layers),
            "score_metric": tag,
            "selection_mode": selection_mode,
            "cutoff_score": cutoff_score,
            "selected_layer_ids": selected_sorted,
            "dropped_layer_ids": dropped,
        }

    def _compute_layer_scores_from_activation_stats(self) -> dict:
        if not self.layer_adaptivity:
            self.allowed_layer_ids = None
            self._log("layer_selection skipped: layer_adaptivity disabled")
            return {"selected_layers": 0, "score_metric": "disabled", "selection_mode": "all"}

        metric = self.layer_selection_metric
        if metric == "fisher":
            # 实现了基于 activation 的 mean/variance 打分；
            self.allowed_layer_ids = None
            self._log("layer_selection skipped: fisher is not implemented in worker path; keep all layers")
            return {"selected_layers": 0, "score_metric": "fisher_unimplemented", "selection_mode": "all"}
        if metric not in ("mean", "variance"):
            self.allowed_layer_ids = None
            self._log("layer_selection skipped: unknown metric=%s; keep all layers", metric)
            return {"selected_layers": 0, "score_metric": metric, "selection_mode": "all"}
        if not self.activation_stats:
            self.allowed_layer_ids = None
            self._log("layer_selection skipped: activation_stats is empty")
            return {"selected_layers": 0, "score_metric": metric, "selection_mode": "all"}

        layer_scores: dict[int, float] = defaultdict(float)
        layer_module_counts: dict[int, int] = defaultdict(int)
        layer_token_counts: dict[int, float] = defaultdict(float)
        num_samples = 0
        for name, stats in self.activation_stats.items():
            layer_idx = self._extract_layer_index(name)
            if layer_idx is None:
                continue
            count = float(stats.get("count", 0.0))
            if count <= 0:
                continue

            mean = stats["sum"].float() / count
            var = (stats["sumsq"].float() / count) - mean.pow(2)
            # 当前支持两种层分数：
            # 1. mean: 该层平均激活强度；
            # 2. variance: 该层激活波动强度。
            # 分数越高，说明该层在 anchor 数据上越“活跃/重要”，越优先保留进 NSP。
            if metric == "mean":
                score = mean.abs().mean().item()
            else:
                score = var.clamp_min(0.0).mean().item()
            layer_scores[layer_idx] += float(score)
            layer_module_counts[layer_idx] += 1
            layer_token_counts[layer_idx] += count
            num_samples += 1

        if layer_scores:
            self._log(
                "layer_scores collected: metric=%s, layers=%d, modules=%d",
                metric,
                len(layer_scores),
                num_samples,
            )
            self._log_detail(
                "layer_scores module_counts: %s",
                {int(layer_id): int(layer_module_counts[layer_id]) for layer_id in sorted(layer_module_counts)},
            )
            self._log_detail(
                "layer_scores token_counts: %s",
                {int(layer_id): float(layer_token_counts[layer_id]) for layer_id in sorted(layer_token_counts)},
            )

        return self._finalize_layer_selection(layer_scores, num_samples, metric, layer_module_counts)

    def _compute_feature_selection_from_activation_stats(self) -> dict:
        # Feature-level 选择:
        # 对每个 eligible module 内的 hidden dimension 单独打分，
        # 只把高分维度交给 optimizer 构造 NSP 子空间，其余维度不加约束。
        if not self.feature_level_adaptivity:
            self.selected_feature_indices = {}
            self._log("feature_selection skipped: feature_level_adaptivity disabled")
            return {
                "selected_feature_modules": 0,
                "selected_feature_dims": 0,
                "total_feature_dims": 0,
                "feature_score_metric": "disabled",
            }
        metric = self.feature_activity_metric
        if metric not in ("variance", "mean_abs"):
            self.selected_feature_indices = {}
            self._log("feature_selection skipped: unknown metric=%s; keep full features", metric)
            return {
                "selected_feature_modules": 0,
                "selected_feature_dims": 0,
                "total_feature_dims": 0,
                "feature_score_metric": metric,
            }
        if not self.activation_stats:
            self.selected_feature_indices = {}
            self._log("feature_selection skipped: activation_stats is empty")
            return {
                "selected_feature_modules": 0,
                "selected_feature_dims": 0,
                "total_feature_dims": 0,
                "feature_score_metric": metric,
            }

        selected_modules = self.to_be_updated_modules
        new_selected: dict[str, torch.Tensor] = {}
        module_summaries = []
        total_selected_dims = 0
        total_feature_dims = 0
        keep_ratio = float(min(max(self.high_freq_block_ratio, 0.0), 1.0))
        min_keep = max(1, int(self.high_freq_min_blocks))

        for name, stats in self.activation_stats.items():
            if name not in selected_modules:
                continue
            count = float(stats.get("count", 0.0))
            if count <= 0:
                continue
            mean = stats["sum"].float() / count
            var = (stats["sumsq"].float() / count) - mean.pow(2)
            # 当前支持两种 feature score：
            # 1. mean_abs: 每个维度的平均激活绝对值；
            # 2. variance: 每个维度在 anchor 数据上的方差。
            # 这对应 README 里的“高重要性特征更细粒度保护”思想。
            scores = mean.abs() if metric == "mean_abs" else var.clamp_min(0.0)
            feat_dim = int(scores.numel())
            if feat_dim <= 0:
                continue

            keep_n = max(min_keep, int(feat_dim * keep_ratio))
            keep_n = max(1, min(feat_dim, keep_n))
            total_feature_dims += feat_dim
            if keep_n >= feat_dim:
                idx = torch.arange(feat_dim, device=scores.device, dtype=torch.long)
            else:
                # 这里做的是按维度分数的 hard top-k。
                # 最终传给 optimizer 的 `selected_feature_indices[name]`
                # 就是该模块参与 eigen / projection 构造的维度子集。
                _, idx = torch.topk(scores, k=keep_n, largest=True, sorted=False)
                idx = idx.sort().values
            new_selected[name] = idx.detach()
            total_selected_dims += int(idx.numel())

            score_min = float(scores.min().item()) if scores.numel() > 0 else 0.0
            score_max = float(scores.max().item()) if scores.numel() > 0 else 0.0
            score_mean = float(scores.mean().item()) if scores.numel() > 0 else 0.0
            module_summaries.append(
                {
                    "name": name,
                    "selected": int(idx.numel()),
                    "total": feat_dim,
                    "score_min": score_min,
                    "score_max": score_max,
                    "score_mean": score_mean,
                    "selected_idx_preview": idx[: min(16, idx.numel())].detach().cpu().tolist(),
                }
            )

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            obj = [
                {
                    name: idx.detach().cpu().tolist()
                    for name, idx in new_selected.items()
                }
            ]
            torch.distributed.broadcast_object_list(obj, src=0)
            new_selected = {
                name: torch.tensor(indices, dtype=torch.long)
                for name, indices in obj[0].items()
            }

        self.selected_feature_indices = new_selected
        self._log(
            "feature_selection done: metric=%s, selected_modules=%d, selected_dims=%d/%d, keep_ratio=%.3f, min_keep=%d",
            metric,
            len(self.selected_feature_indices),
            total_selected_dims,
            total_feature_dims,
            keep_ratio,
            min_keep,
        )
        preview = module_summaries[: min(8, len(module_summaries))]
        self._log_detail("feature_selection preview: %s", preview)
        self._log_trace(
            "feature_selection full: %s",
            module_summaries,
        )
        return {
            "selected_feature_modules": len(self.selected_feature_indices),
            "selected_feature_dims": total_selected_dims,
            "total_feature_dims": total_feature_dims,
            "feature_score_metric": metric,
        }

    def compute_cov(self, module, fea_in, fea_out, name: str | None = None):
        """Forward hook: collect features to build a covariance matrix."""
        if not self.enable_nullspace_projection:
            return None

        if name is None:
            name = getattr(module, "__qualname__", "unknown_module")

        x = fea_in[0]  # (1, valid_tokens, D)
        # IMPORTANT: Since flash-attn is used, no padding tokens, we dont need attention_mask

        try:
            # 将 token 维展平后，把当前层所有有效 token 看作 anchor 表征样本，
            # 用于近似 README 里的 H_l，并累计 H^T H 形式的协方差统计。
            feats = x.reshape(-1, x.shape[-1])  # (valid_tokens, D)
            self._log_trace("compute_cov: module=%s, raw_feat_shape=%s", name, tuple(feats.shape))
            feats = self._hard_select_tokens(feats, name)
            self.update_cov(feats, name)
            self._update_activation_stats(name, feats)
        except Exception:
            # 个别模块输入形状不满足预期时退化为 batch 内平均表征，
            # 这样至少还能得到稳定的层级/特征活跃度统计。
            feats = x.mean(dim=1)  # (N, D)
            self._log_trace("compute_cov fallback_mean: module=%s, raw_feat_shape=%s", name, tuple(feats.shape))
            feats = self._hard_select_tokens(feats, name)
            self.update_cov(feats, name)
            self._update_activation_stats(name, feats)
        return None

    @contextmanager
    def hook_module(self):
        """Context manager to register/unregister NSP forward hooks on eligible modules."""
        if not self.enable_nullspace_projection:
            yield
            return

        handles = []
        try:
            # 只在被 NSP 选中的 attention/MLP 线性层上挂 hook。
            # 这样既对应 README 的“选择性应用”，也避免对全模型收集统计带来的开销。
            for name, m in self.nsp_stat_modules.items():
                hook_fun = functools.partial(self.compute_cov, name=name)
                handles.append(m.register_forward_hook(hook_fun))
            yield
        finally:
            for handle in handles:
                handle.remove()

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_nsp(self, data: DataProto):
        """Run a forward pass with NSP hooks to accumulate covariance on the actor.

        Phase 1 only collects covariance statistics, without updating optimizer
        or projection matrices.
        """
        assert self._is_actor
        if not self.enable_nullspace_projection:
            return DataProto()

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # Align DataProto meta_info with compute_log_prob
        config_source = self.config.rollout
        data.meta_info["micro_batch_size"] = config_source.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = config_source.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = config_source.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        self._log(
            "update_nsp start: batch_size=%d, token_importance=%s, top_ratio=%.3f",
            len(data),
            self.token_importance,
            float(self.token_importance_top_ratio),
        )

        # 核心点是“借用 actor 原有前向路径触发 hook”：
        # NSP 这里只采统计，不参与 loss；真正的投影矩阵会在 trainer 触发的
        # sync/statistics/projection 阶段统一构造并注入 optimizer。
        with self.ulysses_sharding_manager:
            with torch.no_grad():
                with self.hook_module():
                    # re-use actor's compute_log_prob to trigger forward; discard outputs
                    _log_probs, _entropys = self.actor.compute_log_prob(data=data, calculate_entropy=True)

        # unshard FSDP root if needed (same as compute_log_prob)
        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during NSP update", logger=logger)

        self._log(
            "update_nsp done: modules=%d, counts=%d",
            len(self.fea_in),
            len(self.total_num_for_cov_update),
        )
        # Return an empty DataProto as required by the decorator
        return DataProto()

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_nsp_from_replay(self, data: DataProto):
        """Collect NSP covariance statistics from replay-anchor examples.

        Reuses the actor's original micro-batch forward path so remove-padding
        stays consistent with the default NSP collection flow.
        """
        assert self._is_actor
        if not self.enable_nullspace_projection:
            return DataProto()

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        eval_batch_size = int(data.meta_info.get("replay_eval_batch_size", 0) or 0)
        if eval_batch_size <= 0:
            eval_batch_size = len(data)
        self._log(
            "update_nsp_from_replay start: batch_size=%d, eval_batch_size=%d, token_importance=%s, top_ratio=%.3f",
            len(data),
            eval_batch_size,
            self.token_importance,
            float(self.token_importance_top_ratio),
        )

        model = self.actor.actor_module
        was_training = model.training
        model.eval()
        try:
            with self.ulysses_sharding_manager:
                with torch.no_grad():
                    with self.hook_module():
                        micro_batch_count = 0
                        for micro_batch in data.split(eval_batch_size):
                            micro_batch_count += 1
                            micro_batch = micro_batch.to(get_device_id())
                            input_ids = micro_batch.batch["input_ids"]
                            attention_mask = micro_batch.batch["attention_mask"]
                            if input_ids.dim() == 1:
                                input_ids = input_ids.unsqueeze(0)
                            if attention_mask.dim() == 1:
                                attention_mask = attention_mask.unsqueeze(0)
                            position_ids = compute_position_id_with_mask(attention_mask)
                            seq_lens = attention_mask.sum(dim=-1).clamp_min(1).long()
                            response_indices = (seq_lens - 1).unsqueeze(-1)
                            responses = torch.gather(input_ids, dim=1, index=response_indices)
                            # 这里把 replay example 伪装成 actor 原本就能处理的输入结构，
                            # 目的是复用同一条 micro-batch 前向路径，让 remove-padding、
                            # FSDP/unshard、hook 触发位置都与正常 rollout 路径保持一致。
                            model_inputs = {
                                "input_ids": input_ids,
                                "attention_mask": attention_mask,
                                "position_ids": position_ids,
                                "responses": responses,
                            }
                            _ = self.actor._forward_micro_batch(
                                model_inputs,
                                temperature=self.config.rollout.temperature,
                                calculate_entropy=False,
                            )
                        self._log_detail(
                            "update_nsp_from_replay forward_summary: micro_batches=%d, modules=%d, counts=%d",
                            int(micro_batch_count),
                            len(self.fea_in),
                            len(self.total_num_for_cov_update),
                        )
        finally:
            if was_training:
                model.train()

        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during replay-anchor NSP update", logger=logger)

        self._log(
            "update_nsp_from_replay done: modules=%d, counts=%d",
            len(self.fea_in),
            len(self.total_num_for_cov_update),
        )
        return DataProto()

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def encode_replay_embeddings(self, data: DataProto):
        """Encode replay examples into mean-pooled hidden-state embeddings."""
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        eval_batch_size = int(data.meta_info.get("replay_eval_batch_size", 0) or 0)
        if eval_batch_size <= 0:
            eval_batch_size = len(data)
        hidden_layer_index = int(data.meta_info.get("replay_hidden_layer_index", -1))
        return_batch_centroid = bool(data.meta_info.get("return_batch_centroid", False))

        embeddings: list[torch.Tensor] = []
        model = self.actor.actor_module
        was_training = model.training
        model.eval()
        try:
            with self.ulysses_sharding_manager:
                for micro_batch in data.split(eval_batch_size):
                    micro_batch = micro_batch.to(get_device_id())
                    input_ids = micro_batch.batch["input_ids"]
                    attention_mask = micro_batch.batch["attention_mask"]
                    if input_ids.dim() == 1:
                        input_ids = input_ids.unsqueeze(0)
                    if attention_mask.dim() == 1:
                        attention_mask = attention_mask.unsqueeze(0)
                    position_ids = compute_position_id_with_mask(attention_mask)
                    if position_ids.dim() == 1:
                        position_ids = position_ids.unsqueeze(0)
                    with torch.no_grad():
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            output_hidden_states=True,
                            use_cache=False,
                        )
                    hidden_states = outputs.hidden_states
                    if hidden_states is None:
                        raise RuntimeError("Hidden states are required for replay embedding encoding")
                    if hidden_layer_index >= len(hidden_states) or hidden_layer_index < -len(hidden_states):
                        raise IndexError(
                            f"replay_hidden_layer_index={hidden_layer_index} is out of range for {len(hidden_states)} hidden states"
                        )
                    hidden = hidden_states[hidden_layer_index]
                    if hidden.dim() == 2:
                        hidden = hidden.unsqueeze(0)
                    # 这里的 mean-pooling 就是 README 里“候选样本映射到隐藏态表示空间”的实现。
                    # 后面 k-means 聚类、sentinel 选择、prototype 相关性计算都依赖这份 embedding。
                    mask = attention_mask.to(hidden.device, dtype=hidden.dtype).unsqueeze(-1)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                    if pooled.dim() == 1:
                        pooled = pooled.unsqueeze(0)
                    embeddings.append(pooled.detach().cpu())
        finally:
            if was_training:
                model.train()

        embedding_tensor = torch.cat(embeddings, dim=0) if embeddings else torch.empty((0, 0), dtype=torch.float32)
        if embedding_tensor.dim() == 1:
            embedding_tensor = embedding_tensor.unsqueeze(0)
        if return_batch_centroid:
            # DataProto requires all tensor fields to share the same batch dimension.
            # `embeddings` is per-example ([batch, hidden]) while `batch_centroid` is
            # a batch aggregate ([1, hidden]), so they cannot be returned together.
            if embedding_tensor.numel() > 0:
                tensors = {"batch_centroid": embedding_tensor.mean(dim=0, keepdim=True)}
            else:
                tensors = {"batch_centroid": torch.empty((0, 0), dtype=torch.float32)}
        else:
            tensors = {"embeddings": embedding_tensor}
        output = DataProto.from_dict(
            tensors=tensors
        )
        output = output.to("cpu")

        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during replay embedding encode", logger=logger)

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_replay_sft_losses(self, data: DataProto):
        """Compute per-example replay SFT losses for a supplied batch."""
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        eval_batch_size = int(data.meta_info.get("replay_eval_batch_size", 0) or 0)
        if eval_batch_size <= 0:
            eval_batch_size = len(data)

        losses: list[torch.Tensor] = []
        model = self.actor.actor_module
        was_training = model.training
        model.eval()
        try:
            with self.ulysses_sharding_manager:
                for micro_batch in data.split(eval_batch_size):
                    micro_batch = micro_batch.to(get_device_id())
                    input_ids = micro_batch.batch["input_ids"]
                    attention_mask = micro_batch.batch["attention_mask"]
                    labels = micro_batch.batch["labels"]
                    if input_ids.dim() == 1:
                        input_ids = input_ids.unsqueeze(0)
                    if attention_mask.dim() == 1:
                        attention_mask = attention_mask.unsqueeze(0)
                    if labels.dim() == 1:
                        labels = labels.unsqueeze(0)
                    position_ids = compute_position_id_with_mask(attention_mask)
                    if position_ids.dim() == 1:
                        position_ids = position_ids.unsqueeze(0)
                    with torch.no_grad():
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            use_cache=False,
                        )
                    logits = outputs.logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    # 这里算的是每个 replay/sentinel 样本的 token-level CE，再按有效 target token 求平均。
                    # 这份 per-example loss 一部分用于 task start 的 baseline 记录，
                    # 一部分用于训练中的 current loss 评估，两者相减即 Delta Loss。
                    token_loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        shift_labels.view(-1),
                        reduction="none",
                        ignore_index=-100,
                    ).view_as(shift_labels)
                    valid = (shift_labels != -100).to(token_loss.dtype)
                    per_example = (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
                    losses.append(per_example.detach().cpu())
        finally:
            if was_training:
                model.train()

        output = DataProto.from_dict(
            tensors={"replay_sft_losses": torch.cat(losses, dim=0) if losses else torch.empty((0,), dtype=torch.float32)}
        )
        output = output.to("cpu")

        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during replay loss eval", logger=logger)

        return output

    def _sync_covariance_matrices_across_ranks(self):
        """Synchronize covariance matrices across all ranks via all_reduce."""
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return

        # 各 rank 的 hook 只看到了本地 micro-batch，需要先 all-reduce 成全局协方差，
        # 才能让后续特征分解拿到跨卡一致的 anchor 子空间。
        for name, cov in self.fea_in.items():
            torch.distributed.all_reduce(cov, op=torch.distributed.ReduceOp.SUM)
            if name in self.total_num_for_cov_update:
                count_tensor = torch.tensor(
                    [self.total_num_for_cov_update[name]], device=cov.device, dtype=torch.float32
                )
                torch.distributed.all_reduce(count_tensor, op=torch.distributed.ReduceOp.SUM)
                self.total_num_for_cov_update[name] = float(count_tensor.item())

    def _sync_activation_stats_across_ranks(self):
        """Synchronize activation summary statistics across all ranks via all_reduce."""
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return

        for name, stats in self.activation_stats.items():
            sum_feat = stats.get("sum")
            sumsq_feat = stats.get("sumsq")
            if torch.is_tensor(sum_feat):
                torch.distributed.all_reduce(sum_feat, op=torch.distributed.ReduceOp.SUM)
            if torch.is_tensor(sumsq_feat):
                torch.distributed.all_reduce(sumsq_feat, op=torch.distributed.ReduceOp.SUM)
            count_tensor = torch.tensor(
                [float(stats.get("count", 0.0))],
                device=sum_feat.device if torch.is_tensor(sum_feat) else self._get_nsp_target_device(),
                dtype=torch.float32,
            )
            torch.distributed.all_reduce(count_tensor, op=torch.distributed.ReduceOp.SUM)
            stats["count"] = float(count_tensor.item())

    def _nsp_state_path(self, local_path: str) -> str:
        return os.path.join(local_path, f"nsp_state_world_size_{self.world_size}_rank_{self.rank}.pt")

    def _serialize_nsp_state(self) -> dict:
        serialized_fea_in = {}
        for name, cov in self.fea_in.items():
            if torch.is_tensor(cov):
                serialized_fea_in[name] = cov.detach().cpu()

        serialized_activation_stats = {}
        for name, stats in self.activation_stats.items():
            if not isinstance(stats, dict):
                continue
            if "sum" not in stats or "sumsq" not in stats:
                continue
            serialized_activation_stats[name] = {
                "sum": stats["sum"].detach().cpu(),
                "sumsq": stats["sumsq"].detach().cpu(),
                "count": float(stats.get("count", 0.0)),
            }

        serialized_total_num = {
            name: float(value)
            for name, value in self.total_num_for_cov_update.items()
        }
        return {
            "fea_in": serialized_fea_in,
            "total_num_for_cov_update": serialized_total_num,
            "activation_stats": serialized_activation_stats,
        }

    def _get_nsp_target_device(self) -> torch.device:
        if device_name == "cpu":
            return torch.device("cpu")
        return torch.device(device_name, get_device_id())

    def _restore_nsp_state(self, payload: dict | None) -> None:
        self.fea_in = defaultdict(dict)
        self.total_num_for_cov_update = {}
        self.activation_stats = defaultdict(dict)
        self.allowed_layer_ids = None
        self.selected_feature_indices = {}

        if not payload:
            return

        target_device = self._get_nsp_target_device()
        for name, cov in (payload.get("fea_in", {}) or {}).items():
            if not torch.is_tensor(cov):
                continue
            self.fea_in[name] = cov.to(device=target_device)

        for name, value in (payload.get("total_num_for_cov_update", {}) or {}).items():
            self.total_num_for_cov_update[name] = float(value)

        for name, stats in (payload.get("activation_stats", {}) or {}).items():
            if not isinstance(stats, dict):
                continue
            sum_feat = stats.get("sum")
            sumsq_feat = stats.get("sumsq")
            if not torch.is_tensor(sum_feat) or not torch.is_tensor(sumsq_feat):
                continue
            self.activation_stats[name] = {
                "sum": sum_feat.to(device=target_device),
                "sumsq": sumsq_feat.to(device=target_device),
                "count": float(stats.get("count", 0.0)),
            }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        super().save_checkpoint(
            local_path=local_path,
            hdfs_path=hdfs_path,
            global_step=global_step,
            max_ckpt_to_keep=max_ckpt_to_keep,
        )

        if local_path is None:
            return

        os.makedirs(local_path, exist_ok=True)
        state_path = self._nsp_state_path(local_path)
        torch.save(self._serialize_nsp_state(), state_path)
        print(f"[NSP] Saved stats checkpoint to {state_path}")

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        super().load_checkpoint(
            local_path=local_path,
            hdfs_path=hdfs_path,
            del_local_after_load=del_local_after_load,
        )

        self._skip_next_nsp_reset_after_resume = False
        self._restore_nsp_state(None)

        if local_path is None:
            return

        state_path = self._nsp_state_path(local_path)
        if not os.path.exists(state_path):
            print(f"[NSP] No stats checkpoint found at {state_path}; starting with empty cache")
            return

        payload = torch.load(state_path, map_location="cpu", weights_only=False)
        self._restore_nsp_state(payload)

        has_covariances = bool(self.fea_in)
        has_counts = any(float(value) != 0.0 for value in self.total_num_for_cov_update.values())
        self._skip_next_nsp_reset_after_resume = has_covariances or has_counts
        print(
            f"[NSP] Loaded stats checkpoint from {state_path}; "
            f"modules={len(self.fea_in)}, resume_skip_reset={self._skip_next_nsp_reset_after_resume}"
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def compute_nsp_statistics(self):
        """Compute activation-based layer/feature selection before projection updates.

        这一步是 README 里“layer level + feature level adaptivity”的集中落点：
        1. 先基于 activation_stats 选层，决定哪些 layer 会启用 NSP；
        2. 再基于 activation_stats 选维度，决定这些层里哪些 hidden features
           真正参与后续的 eigens / transforms 构造。
        """
        assert self._is_actor
        if not self.enable_nullspace_projection:
            return None

        summary = self._compute_layer_scores_from_activation_stats()
        summary["activation_modules"] = len(self.activation_stats)
        feature_summary = self._compute_feature_selection_from_activation_stats()
        summary.update(feature_summary)
        return summary

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def sync_nsp_covariance(self):
        """Synchronize covariance matrices across all ranks.

        This should be called before computing projection matrices (Phase 2) to ensure
        that all ranks have the same global covariance statistics.

        Returns:
            Summary dict with covariance sync information.
        """
        assert self._is_actor
        if not self.enable_nullspace_projection:
            return None

        # Synchronize covariance matrices
        # `fea_in` 供 projection 构造使用，`activation_stats` 供 layer/feature 选择使用，
        # 两者都必须先同步，才能保证所有 rank 上筛出的层和特征一致。
        self._sync_covariance_matrices_across_ranks()
        self._sync_activation_stats_across_ranks()

        # Return summary for logging
        summary = {
            "num_modules_synced": len(self.fea_in),
            "total_covariances_synced": len(self.fea_in),
            "activation_modules_synced": len(self.activation_stats),
        }
        return summary

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset_nsp_cache(self):
        """Reset cached covariance statistics."""
        assert self._is_actor
        if not self.enable_nullspace_projection:
            print("[NSP] reset_nsp_cache skipped: null-space projection disabled")
            return None

        if self._skip_next_nsp_reset_after_resume:
            self._skip_next_nsp_reset_after_resume = False
            print("[NSP] Skip one reset after checkpoint resume to preserve cached stats")
            return None

        print(
            "[NSP] reset_nsp_cache: "
            f"modules={len(self.fea_in)}, "
            f"counts={len(self.total_num_for_cov_update)}, "
            f"activation_modules={len(self.activation_stats)}, "
            f"allowed_layers={0 if self.allowed_layer_ids is None else len(self.allowed_layer_ids)}, "
            f"feature_selected_modules={len(self.selected_feature_indices)}"
        )
        self.fea_in = defaultdict(dict)
        self.total_num_for_cov_update = {}
        self.activation_stats = defaultdict(dict)
        self.allowed_layer_ids = None
        self.selected_feature_indices = {}
        if device_name == "npu":
            self._log("reset_nsp_cache: skip aggressive_empty_cache on NPU for stability")
        else:
            aggressive_empty_cache()
        print("[NSP] reset_nsp_cache done")

        return None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def compute_nsp_projections(self):
        """Compute NSP projection matrices from accumulated covariance and update optimizer.

        This method:
        1. Transfers covariance state to the optimizer
        2. Calls optimizer.get_eigens() to compute eigenvalues/eigenvectors
        3. Calls optimizer.get_transforms() to compute projection matrices

        The optimizer will automatically apply projections in subsequent optimizer.step() calls.

        Returns:
            Summary dict with projection computation information.
        """

        assert self._is_actor
        if not self.enable_nullspace_projection:
            return None

        # Check if optimizer is AdamW_null_space
        from recipe.continual_ns_grpo.ns_adamw import AdamW_null_space
        if not isinstance(self.actor_optimizer, AdamW_null_space):
            print(f"[NSP Warning] optimizer is not AdamW_null_space, skipping projection update")
            return {"error": "optimizer_type_mismatch"}

        self.actor_optimizer.transforms = {}
        self.actor_optimizer.soft_projection_alpha = float(
            min(max(getattr(self.nsp_cfg, "soft_projection_alpha", 0.0) or 0.0, 0.0), 1.0)
        )

        selected_modules = self.to_be_updated_modules
        # `selected_modules` 已经隐含 layer-level 过滤结果；
        # `selected_feature_indices` 则携带 feature-level 过滤结果。
        # optimizer 会用:
        #   covariance + selected modules + selected feature dims
        # 三者共同构造最终 null-space projection。
        self._log(
            "compute_nsp_projections: selected_modules=%d, allowed_layers=%s, activation_modules=%d, feature_selected_modules=%d, soft_projection_alpha=%.3f",
            len(selected_modules),
            "all" if self.allowed_layer_ids is None else len(self.allowed_layer_ids),
            len(self.activation_stats),
            len(self.selected_feature_indices),
            float(self.actor_optimizer.soft_projection_alpha),
        )
        if self.allowed_layer_ids is not None:
            self._log_detail(
                "compute_nsp_projections selected_layer_ids: %s",
                sorted(int(x) for x in self.allowed_layer_ids),
            )
        if self.selected_feature_indices:
            feature_preview = {
                name: idx[: min(16, idx.numel())].detach().cpu().tolist()
                for name, idx in list(self.selected_feature_indices.items())[: min(8, len(self.selected_feature_indices))]
            }
            self._log_detail("compute_nsp_projections selected_feature_preview: %s", feature_preview)
        print(f"[NSP] Computing eigenvalues from {len(self.fea_in)} modules...")
        # get_eigens 会先在筛过的维度子空间上做特征分解，
        # 然后 get_transforms 再把这些结果转成 optimizer.step() 可直接使用的投影矩阵。
        self.actor_optimizer.get_eigens(self.fea_in, selected_modules, self.selected_feature_indices)

        print(f"[NSP] Computing projection matrices...")
        self.actor_optimizer.get_transforms()

        num_params_with_projection = len(self.actor_optimizer.transforms)
        summary = {
            "num_params_with_projection": num_params_with_projection,
            "soft_projection_alpha": float(self.actor_optimizer.soft_projection_alpha),
        }
        print(f"[NSP] Projection matrices ready: {summary}")
        if device_name == "npu":
            self._log("compute_nsp_projections: skip aggressive_empty_cache on NPU for stability")
        else:
            aggressive_empty_cache()
        return summary
