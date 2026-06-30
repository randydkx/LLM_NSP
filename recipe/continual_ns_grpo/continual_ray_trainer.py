import json
import math
import os
import random
import uuid
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from pprint import pprint
from tqdm import tqdm
from omegaconf import OmegaConf
from collections import defaultdict

from verl.utils.tracking import Tracking
from verl.utils.rollout_skip import RolloutSkip
from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import compute_response_mask
from verl.utils.debug import marked_timer
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, compute_reward_async
from verl.trainer.ppo.ray_trainer import compute_advantage,apply_kl_penalty
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.utils import Role
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.trainer.ppo.metric_utils import (compute_data_metrics, compute_timing_metrics,
                                        compute_throughout_metrics)
from verl.utils.metric import reduce_metrics
import ray

from dataclasses import dataclass, fields as dataclass_fields
from typing import Any, Optional

from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
from torchdata.stateful_dataloader import StatefulDataLoader

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from recipe.continual_ns_grpo.nsp_config import NullSpaceProjConfig
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path


@dataclass
class ReplayExample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    prompt_text: str
    target_text: str
    task_id: int

class ContinualPPOTrainer(RayPPOTrainer):
    """
    A Ray PPO trainer for continual learning tasks.
    Takes train_dataset as a list.
    """

    def __init__(self, *args, **kwargs):
        """Initialize continual PPO trainer and setup NSP-related state."""
        super().__init__(*args, **kwargs)

        # NSP configuration (optional)
        # Note: NSP config should be under actor_rollout_ref.nsp, not trainer.nsp
        raw_nsp_cfg = None
        if hasattr(self.config, "actor_rollout_ref"):
            raw_nsp_cfg = getattr(self.config.actor_rollout_ref, "nsp", None)

        if raw_nsp_cfg is not None:
            cfg_dict = OmegaConf.to_container(raw_nsp_cfg, resolve=True)
            allowed_keys = {f.name for f in dataclass_fields(NullSpaceProjConfig)}
            filtered = {k: v for k, v in cfg_dict.items() if k in allowed_keys}
            self.nsp_cfg = NullSpaceProjConfig(**filtered)
        else:
            self.nsp_cfg = NullSpaceProjConfig()

        self.nsp_enabled = bool(self.nsp_cfg.enable_nullspace_projection)
        self.nsp_update_interval = int(self.nsp_cfg.update_projections_every)
        # Reuse legacy NSP switch: when False, skip block/layer statistics selection.
        self.nsp_use_statistics = bool(getattr(self.nsp_cfg, "do_update_statistics", True))
        self.nsp_projection_mode = str(getattr(self.nsp_cfg, "projection_update_mode", "periodic") or "periodic").lower()
        self.nsp_cov_update_every = int(
            getattr(self.nsp_cfg, "cov_update_every", self.nsp_update_interval or 1) or 1
        )
        self.nsp_reset_on_task_start = bool(getattr(self.nsp_cfg, "reset_stats_on_task_start", True))
        self.nsp_anchor_task_id = int(getattr(self.nsp_cfg, "anchor_task_id", 0) or 0)
        self.nsp_anchor_update_interval = int(getattr(self.nsp_cfg, "anchor_update_interval", 0) or 0)
        self.nsp_anchor_max_examples = int(getattr(self.nsp_cfg, "anchor_max_examples", 0) or 0)
        self.nsp_anchor_batch_size = max(1, int(getattr(self.nsp_cfg, "anchor_batch_size", 4) or 4))
        self.nsp_anchor_shuffle = bool(getattr(self.nsp_cfg, "anchor_shuffle", True))
        self.nsp_refresh_first_step = bool(getattr(self.nsp_cfg, "should_update_NSP_first_step", True))
        self.nsp_last_refresh_step = 0
        self.nsp_refresh_count = 0
        # Placeholder for future BlockNSPWrapper integration
        self.nsp_wrapper = None

        raw_replay_cfg = None
        if hasattr(self.config, "actor_rollout_ref"):
            raw_replay_cfg = getattr(self.config.actor_rollout_ref, "replay", None)
        replay_cfg = OmegaConf.to_container(raw_replay_cfg, resolve=True) if raw_replay_cfg is not None else {}
        if replay_cfg is None:
            replay_cfg = {}
        print(replay_cfg)

        self.enable_interference_recall = bool(replay_cfg.get("enable_interference_recall", False))
        self.sentinel_selection_max_examples = int(replay_cfg.get("sentinel_selection_max_examples", 0) or 0)
        self.sentinel_candidate_cap = int(replay_cfg.get("sentinel_candidate_cap", 0) or 0)
        self.sentinel_sample_count = int(replay_cfg.get("sentinel_sample_count", 0) or 0)
        self.selected_cluster_num_k = int(replay_cfg.get("selected_cluster_num_k", 0) or 0)
        self.num_replayed_examples_per_cluster = int(replay_cfg.get("num_replayed_examples_per_cluster", 0) or 0)
        self.sft_replay_batch_size = max(1, int(replay_cfg.get("sft_replay_batch_size", 1) or 1))
        self.replay_loss_lambda = float(replay_cfg.get("replay_loss_lambda", 0.0) or 0.0)
        self.sft_replay_start_decay_step = int(replay_cfg.get("start_decay_step", 20) or 20)
        self.sentinel_eval_interval = max(1, int(replay_cfg.get("sentinel_eval_interval", 1) or 1))
        self.sentinel_hidden_state_batch_size = max(1, int(replay_cfg.get("sentinel_hidden_state_batch_size", 4) or 4))
        self.sentinel_sft_max_seq_length = int(
            replay_cfg.get("sentinel_sft_max_seq_length", self.config.data.get("max_prompt_length", 1024) + 512) or 1536
        )
        self.sft_target_key = replay_cfg.get("sft_target_key", None)
        self.sentinel_shuffle_candidates = bool(replay_cfg.get("sentinel_shuffle_candidates", True))
        self.sentinel_kmeans_iters = max(1, int(replay_cfg.get("sentinel_kmeans_iters", 10) or 10))
        self.replay_score_alpha = float(replay_cfg.get("score_alpha", 1.0))
        self.replay_score_beta = float(replay_cfg.get("score_beta", 0.0))
        self.prototype_hidden_layer_index = int(replay_cfg.get("prototype_hidden_layer_index", -2))
        self.prototype_ema_decay = float(replay_cfg.get("prototype_ema_decay", 0.95))
        self.prototype_update_interval = max(1, int(replay_cfg.get("prototype_update_interval", 1)))
        self.prototype_task_ids = self._normalize_replay_task_ids(replay_cfg.get("prototype_task_ids", []))
        raw_seed_jsonl_paths = replay_cfg.get("seed_jsonl_path_list", None)
        print("raw seed jsonl paths:", raw_seed_jsonl_paths)
        if raw_seed_jsonl_paths is None:
            legacy_seed_jsonl_path = replay_cfg.get("seed_jsonl_path", None)
            raw_seed_jsonl_paths = [legacy_seed_jsonl_path] if legacy_seed_jsonl_path else []
        elif isinstance(raw_seed_jsonl_paths, str):
            raw_seed_jsonl_paths = [raw_seed_jsonl_paths]
        else:
            raw_seed_jsonl_paths = [str(path) for path in list(raw_seed_jsonl_paths) if path]
        self.seed_jsonl_path_list = [str(path) for path in raw_seed_jsonl_paths if path]
        self.seed_jsonl_max_examples_per_file = int(replay_cfg.get("seed_jsonl_max_examples_per_file", 0) or 0)
        self.replay_ingest_task_candidates = bool(replay_cfg.get("ingest_task_replay_candidates", True))

        replay_seed = int(self.config.data.get("seed", 0) or 0)
        self._replay_rng = random.Random(replay_seed)
        self._default_sft_target_keys = ["generation", "answer", "reference"]
        self._sentinel_candidate_examples: list[dict[str, Any]] = []
        self._sentinel_active_pool: Optional[dict[str, Any]] = None
        self._tasks_with_candidates: set[int] = set()
        self._replay_resume_checkpoint_path: Optional[str] = None
        self._allow_replay_on_first_task = False
        self._code_prototype: Optional[torch.Tensor] = None
        self._code_prototype_initialized = False
        self._code_prototype_source_task: Optional[int] = None
        print(
            "[Replay] "
            f"enabled={self.enable_interference_recall}, "
            f"lambda={self.replay_loss_lambda}, "
            f"candidate_cap={self.sentinel_candidate_cap}, "
            f"sentinel_sample_count={self.sentinel_sample_count}, "
            f"selected_cluster_num_k={self.selected_cluster_num_k}, "
            f"examples_per_cluster={self.num_replayed_examples_per_cluster}, "
            f"score_alpha={self.replay_score_alpha}, "
            f"score_beta={self.replay_score_beta}, "
            f"prototype_layer={self.prototype_hidden_layer_index}, "
            f"prototype_tasks={sorted(self.prototype_task_ids)}, "
            f"seed_jsonl_files={len(self.seed_jsonl_path_list)}"
        )

    @staticmethod
    def _normalize_replay_task_ids(raw_task_ids) -> set[int]:
        if raw_task_ids is None:
            return set()
        if isinstance(raw_task_ids, (int, np.integer)):
            values = [int(raw_task_ids)]
        else:
            try:
                values = [int(task_id) for task_id in list(raw_task_ids)]
            except Exception:
                return set()

        values = [task_id for task_id in values if task_id >= 0]
        if values and 0 not in values and all(task_id >= 1 for task_id in values):
            values = [task_id - 1 for task_id in values]
        return set(values)

    def _is_code_prototype_task(self, task_idx: int) -> bool:
        return int(task_idx) in self.prototype_task_ids

    def _resolve_task_epochs(self) -> list[int]:
        """Resolve per-task epoch counts with validation and fallback behavior."""
        num_tasks = int(self.config.data.num_tasks)
        raw_task_epochs = getattr(self.config.trainer, "task_epochs", None)

        if raw_task_epochs is None:
            total_epochs = getattr(self.config.trainer, "total_epochs", None)
            if isinstance(total_epochs, bool) or not isinstance(total_epochs, int) or total_epochs <= 0:
                raise ValueError(
                    f"trainer.total_epochs must be a positive integer when trainer.task_epochs is not set, "
                    f"but got {total_epochs!r}"
                )
            return [int(total_epochs)] * num_tasks

        task_epochs = list(raw_task_epochs)
        if len(task_epochs) != num_tasks:
            raise ValueError(
                f"trainer.task_epochs length ({len(task_epochs)}) must match num_tasks ({num_tasks})"
            )

        resolved_task_epochs: list[int] = []
        for task_idx, epoch_count in enumerate(task_epochs):
            if isinstance(epoch_count, bool) or not isinstance(epoch_count, int):
                raise ValueError(
                    "trainer.task_epochs must contain positive integers only, "
                    f"but task {task_idx + 1} has value {epoch_count!r}"
                )
            if epoch_count <= 0:
                raise ValueError(
                    "trainer.task_epochs must contain positive integers only, "
                    f"but task {task_idx + 1} has value {epoch_count}"
                )
            resolved_task_epochs.append(int(epoch_count))

        return resolved_task_epochs

    def _resolve_raw_task_steps(self) -> Optional[list[int]]:
        """Resolve optional per-task step budgets.

        Values must either be positive integers or -1, where -1 means
        "train exactly one epoch" for that task.
        """
        num_tasks = int(self.config.data.num_tasks)
        raw_task_steps = getattr(self.config.trainer, "task_steps", None)
        if raw_task_steps is None:
            return None

        task_steps = list(raw_task_steps)
        if len(task_steps) != num_tasks:
            raise ValueError(
                f"trainer.task_steps length ({len(task_steps)}) must match num_tasks ({num_tasks})"
            )

        resolved_task_steps: list[int] = []
        for task_idx, step_count in enumerate(task_steps):
            if isinstance(step_count, bool) or not isinstance(step_count, int):
                raise ValueError(
                    "trainer.task_steps must contain integers only, "
                    f"but task {task_idx + 1} has value {step_count!r}"
                )
            if step_count == -1:
                resolved_task_steps.append(-1)
                continue
            if step_count <= 0:
                raise ValueError(
                    "trainer.task_steps must contain positive integers or -1 only, "
                    f"but task {task_idx + 1} has value {step_count}"
                )
            resolved_task_steps.append(int(step_count))

        return resolved_task_steps

    def _resolve_task_step_budgets(self) -> list[int]:
        """Resolve actual per-task optimizer-step budgets."""
        if len(self.train_dataloader_list) != int(self.config.data.num_tasks):
            raise ValueError("Task dataloaders must be initialized before resolving task step budgets")

        if self.raw_task_steps is None:
            return [
                len(train_dataloader) * self.task_epochs[task_idx]
                for task_idx, train_dataloader in enumerate(self.train_dataloader_list)
            ]

        resolved_task_steps: list[int] = []
        for task_idx, configured_steps in enumerate(self.raw_task_steps):
            if configured_steps == -1:
                resolved_task_steps.append(len(self.train_dataloader_list[task_idx]))
            else:
                resolved_task_steps.append(int(configured_steps))
        return resolved_task_steps

    def _create_dataloader(self, train_dataset:list, val_dataset:list, collate_fn, train_sampler: list[Sampler]):
        """
        Creates the train and validation dataloaders in continual learning settings.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        print(f"Creating train and validation dataloader for {self.config.data.num_tasks} tasks")
        self.train_dataset_list, self.val_dataset_list = train_dataset, val_dataset

        # train_sampler should be provided as a list from the caller (main function)
        assert train_sampler is not None and isinstance(train_sampler, list), \
            "train_sampler must be provided as a list for continual learning"
        assert len(train_sampler) == self.config.data.num_tasks, \
            f"train_sampler list length ({len(train_sampler)}) must match num_tasks ({self.config.data.num_tasks})"
        
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]
        self.task_epochs = self._resolve_task_epochs()
        self.raw_task_steps = self._resolve_raw_task_steps()
        print(f"Resolved task epochs: {self.task_epochs}")
        if self.raw_task_steps is not None:
            print(f"Resolved raw task steps: {self.raw_task_steps}")

        self.train_dataloader_list = []
        self.val_dataloader_list = []
    
        for task_idx in range(self.config.data.num_tasks):
            train_dataset = self.train_dataset_list[task_idx]
            val_dataset = self.val_dataset_list[task_idx]
            val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
            if val_batch_size is None:
                val_batch_size = len(val_dataset)
            train_dataloader = StatefulDataLoader(
                dataset=train_dataset,
                batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
                num_workers=num_workers,
                drop_last=True,
                collate_fn=collate_fn,
                sampler=train_sampler[task_idx],
            )
            val_dataloader = StatefulDataLoader(
                dataset=val_dataset,
                batch_size=val_batch_size,
                num_workers=num_workers,
                shuffle=self.config.data.get("validation_shuffle", True),
                drop_last=False,
                collate_fn=collate_fn,
            )
            assert len(train_dataloader) >= 1, f"Task {task_idx + 1}: Train dataloader is empty!"
            assert len(val_dataloader) >= 1, f"Task {task_idx + 1}: Validation dataloader is empty!"

            print(
                f"Task {task_idx + 1}/{self.config.data.num_tasks}: "
                f"train_dataloader={len(train_dataloader)}, val_dataloader={len(val_dataloader)}"
            )
            self.train_dataloader_list.append(train_dataloader)
            self.val_dataloader_list.append(val_dataloader)

        self.task_step_budgets = self._resolve_task_step_budgets()
        print(f"Resolved task step budgets: {self.task_step_budgets}")

        total_training_steps = sum(self.task_step_budgets)

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _maybe_update_nsp(
        self,
        batch: DataProto,
        task_idx: int,
        epoch: int,
        metrics: dict[str, Any],
        task_step: Optional[int] = None,
    ) -> None:
        """Trigger NSP refresh/stat accumulation according to the selected mode.

        对应 README 里的 trainer 侧调度职责：
        1. `task_end` 模式下，在 anchor task 训练过程中累计协方差；
        2. `dynamic` 模式下，不依赖固定 anchor task，而是定期从 replay
           candidate 中抽样一批“历史锚点”样本，重新构造投影矩阵。
        """
        if not getattr(self, "nsp_enabled", False):
            return
        if self._should_use_replay_anchor_for_nsp() and self.nsp_projection_mode != "dynamic":
            return

        if self.nsp_projection_mode == "dynamic":
            if not self._should_use_replay_anchor_for_nsp():
                return
            if not self._should_run_dynamic_nsp_refresh(task_step=task_step):
                return
            if self._refresh_nsp_from_replay_anchor(task_idx=task_idx, epoch=epoch, metrics=metrics):
                self.nsp_last_refresh_step = int(task_step or 0)
                metrics["nsp/dynamic_refresh"] = 1
                metrics["nsp/dynamic_refresh_task_step"] = int(task_step or 0)
                return
            print(
                f"[NSP] Skip dynamic replay-anchor refresh at task {task_idx + 1}, "
                f"task_step={int(task_step or 0)}: no replay anchor examples available"
            )
            return

        if getattr(self, "nsp_projection_mode", "task_end") != "task_end":
            return
        if task_idx != self.nsp_anchor_task_id:
            return
        if self.nsp_cov_update_every <= 0:
            return
        if task_step is None:
            return
        if task_step % self.nsp_cov_update_every != 0:
            return

        print(f"[NSP] Accumulating cov stats for anchor task {task_idx + 1} at task step {task_step}")
        self.actor_rollout_wg.update_nsp(batch)
        metrics["nsp/cov_update"] = 1
        metrics["nsp/cov_update_task_id"] = task_idx + 1
        metrics["nsp/cov_update_task_step"] = task_step
        metrics["nsp/cov_update_epoch"] = epoch

    def _should_use_replay_anchor_for_nsp(self) -> bool:
        return bool(
            getattr(self, "nsp_enabled", False)
            and getattr(self, "nsp_projection_mode", "task_end") in ("task_end", "dynamic")
            and self.enable_interference_recall
        )

    def _should_run_dynamic_nsp_refresh(self, task_step: Optional[int]) -> bool:
        if self.nsp_projection_mode != "dynamic":
            return False
        if task_step is None:
            return False
        current_step = int(task_step)
        if current_step <= 0:
            return False
        if self.nsp_refresh_first_step and current_step == 1:
            if int(getattr(self, "nsp_last_refresh_step", -1)) == 0:
                return False
            return True

        interval = int(self.nsp_anchor_update_interval or 0)
        if interval <= 0:
            interval = int(self.nsp_update_interval or 0)
        if interval <= 0:
            return False
        return current_step % interval == 0

    def _get_replay_anchor_examples_for_nsp(self) -> list[ReplayExample]:
        if not self._should_use_replay_anchor_for_nsp():
            return []

        examples = [
            entry["example"]
            for entry in self._sentinel_candidate_examples
            if isinstance(entry, dict) and entry.get("example") is not None
        ]
        if not examples:
            return []

        if self.nsp_anchor_shuffle:
            indices = list(range(len(examples)))
            self._replay_rng.shuffle(indices)
            examples = [examples[idx] for idx in indices]

        if self.nsp_anchor_max_examples > 0:
            examples = examples[: min(len(examples), self.nsp_anchor_max_examples)]

        return examples

    def _refresh_nsp_from_replay_anchor(self, task_idx: int, epoch: int, metrics: dict[str, Any]) -> bool:
        anchor_examples = self._get_replay_anchor_examples_for_nsp()
        if not anchor_examples:
            return False

        batch_size = max(1, int(self.nsp_anchor_batch_size))
        total_batches = math.ceil(len(anchor_examples) / batch_size)
        print(
            "[NSP] Refreshing projections from replay anchor: "
            f"task={task_idx + 1}, examples={len(anchor_examples)}, batch_size={batch_size}, batches={total_batches}"
        )

        # 这一轮 refresh 的完整链路是：
        # replay anchor examples -> worker 前向收集协方差/激活统计 ->
        # 多卡同步统计量 -> 基于统计量筛层/筛特征 -> 计算最终投影矩阵。
        self.actor_rollout_wg.reset_nsp_cache()
        for batch_start in range(0, len(anchor_examples), batch_size):
            batch_examples = anchor_examples[batch_start : batch_start + batch_size]
            proto = self._build_replay_dataproto(batch_examples, batch_size)
            self.actor_rollout_wg.update_nsp_from_replay(proto)

        sync_summary = self.actor_rollout_wg.sync_nsp_covariance()
        stats_summary = self.actor_rollout_wg.compute_nsp_statistics()
        proj_summary = self.actor_rollout_wg.compute_nsp_projections()
        self.actor_rollout_wg.reset_nsp_cache()

        self.nsp_refresh_count += 1
        metrics["nsp/task_end_refresh"] = 1
        metrics["nsp/task_end_task_id"] = task_idx + 1
        metrics["nsp/task_end_epoch"] = epoch
        metrics["nsp/replay_anchor_refresh"] = 1
        metrics["nsp/replay_anchor_examples"] = len(anchor_examples)
        metrics["nsp/replay_anchor_batches"] = total_batches
        metrics["nsp/replay_anchor_candidate_pool"] = len(self._sentinel_candidate_examples)
        metrics["nsp/replay_anchor_refresh_count"] = self.nsp_refresh_count

        if sync_summary:
            metrics["nsp/num_modules_synced"] = sync_summary[0].get("num_modules_synced", 0)
            metrics["nsp/total_covariances_synced"] = sync_summary[0].get("total_covariances_synced", 0)

        if stats_summary and "error" not in stats_summary[0]:
            metrics["nsp/activation_modules"] = stats_summary[0].get("activation_modules", 0)
            metrics["nsp/layer_selection_selected_layers"] = stats_summary[0].get("selected_layers", 0)
            metrics["nsp/layer_selection_total_layers"] = stats_summary[0].get("total_layers_scored", 0)
            metrics["nsp/feature_selection_modules"] = stats_summary[0].get("selected_feature_modules", 0)
            metrics["nsp/feature_selection_selected_dims"] = stats_summary[0].get("selected_feature_dims", 0)
            metrics["nsp/feature_selection_total_dims"] = stats_summary[0].get("total_feature_dims", 0)

        if proj_summary and "error" not in proj_summary[0]:
            metrics["nsp/num_params_with_projection"] = proj_summary[0].get("num_params_with_projection", 0)
            metrics["nsp/projection_anchor_task_id"] = task_idx + 1
            metrics["nsp/soft_projection_alpha"] = proj_summary[0].get("soft_projection_alpha", 0.0)

        return True

    def _finalize_nsp_task_end(self, task_idx: int, epoch: int, metrics: dict[str, Any]) -> None:
        """Finalize NSP once after the anchor task: sync stats and compute projections."""
        if not getattr(self, "nsp_enabled", False):
            return
        if getattr(self, "nsp_projection_mode", "task_end") != "task_end":
            return
        if not self._should_use_replay_anchor_for_nsp() and task_idx != self.nsp_anchor_task_id:
            return

        if self._refresh_nsp_from_replay_anchor(task_idx=task_idx, epoch=epoch, metrics=metrics):
            return
        if self._should_use_replay_anchor_for_nsp():
            print(f"[NSP] Skip replay-anchor refresh at task {task_idx + 1}: no replay anchor examples available")
            return

        print(f"[NSP] Finalizing NSP for anchor task {task_idx + 1}")

        print(f"[NSP] Task-end Step 1/2: Synchronizing covariance matrices across ranks...")
        sync_summary = self.actor_rollout_wg.sync_nsp_covariance()
        print(f"[NSP] Covariance synced: {sync_summary}")

        print(f"[NSP] Task-end Step 2/3: Computing NSP statistics...")
        stats_summary = self.actor_rollout_wg.compute_nsp_statistics()
        print(f"[NSP] Statistics computed: {stats_summary}")

        print(f"[NSP] Task-end Step 3/3: Computing projection matrices...")
        proj_summary = self.actor_rollout_wg.compute_nsp_projections()
        print(f"[NSP] Projections computed: {proj_summary}")

        print(f"[NSP] Task-end Cleanup: Resetting covariance cache...")
        self.actor_rollout_wg.reset_nsp_cache()
        metrics["nsp/task_end_cache_reset"] = 1

        metrics["nsp/task_end_refresh"] = 1
        metrics["nsp/task_end_task_id"] = task_idx + 1
        metrics["nsp/task_end_epoch"] = epoch

        if sync_summary:
            metrics["nsp/num_modules_synced"] = sync_summary[0].get("num_modules_synced", 0)
            metrics["nsp/total_covariances_synced"] = sync_summary[0].get("total_covariances_synced", 0)

        if stats_summary and "error" not in stats_summary[0]:
            metrics["nsp/activation_modules"] = stats_summary[0].get("activation_modules", 0)
            metrics["nsp/layer_selection_selected_layers"] = stats_summary[0].get("selected_layers", 0)
            metrics["nsp/layer_selection_total_layers"] = stats_summary[0].get("total_layers_scored", 0)
            metrics["nsp/feature_selection_modules"] = stats_summary[0].get("selected_feature_modules", 0)
            metrics["nsp/feature_selection_selected_dims"] = stats_summary[0].get("selected_feature_dims", 0)
            metrics["nsp/feature_selection_total_dims"] = stats_summary[0].get("total_feature_dims", 0)

        if proj_summary and "error" not in proj_summary[0]:
            metrics["nsp/num_params_with_projection"] = proj_summary[0].get("num_params_with_projection", 0)
            metrics["nsp/projection_anchor_task_id"] = task_idx + 1
            metrics["nsp/soft_projection_alpha"] = proj_summary[0].get("soft_projection_alpha", 0.0)

    def _get_scheduled_replay_loss_lambda(self, task_step: int, task_total_steps: int) -> float:
        base_lambda = float(getattr(self, "replay_loss_lambda", 0.0) or 0.0)
        if base_lambda <= 0.0:
            return 0.0
        if task_total_steps <= 0:
            return base_lambda
        if task_step < self.sft_replay_start_decay_step:
            return base_lambda
        max_step_index = max(int(task_total_steps) - 1, 0)
        if max_step_index <= self.sft_replay_start_decay_step:
            return 0.0
        progress = (min(int(task_step), max_step_index) - self.sft_replay_start_decay_step) / float(
            max_step_index - self.sft_replay_start_decay_step
        )
        return float(max(0.0, min(base_lambda, base_lambda * (1.0 - progress))))

    def _resolve_prompt_text(self, sample) -> Optional[str]:
        if not isinstance(sample, dict):
            return None
        prompt_value = sample.get("prompt")
        if not isinstance(prompt_value, str):
            prompt_value = sample.get("prompt_text")
        if isinstance(prompt_value, str):
            return prompt_value

        messages = sample.get("messages", prompt_value)
        if isinstance(messages, list) and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                return None
        return None

    def _resolve_target_text(self, sample) -> Optional[str]:
        if not isinstance(sample, dict):
            return None
        candidate_keys: list[str] = []
        if self.sft_target_key:
            candidate_keys.append(str(self.sft_target_key))
        candidate_keys.extend(self._default_sft_target_keys)
        seen: set[str] = set()
        for key in candidate_keys:
            if key in seen:
                continue
            seen.add(key)
            value = sample.get(key)
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
                return "\n".join(value)
        for alias in ("target_text", "answer_text"):
            value = sample.get(alias)
            if isinstance(value, str) and value.strip():
                return value
        return None

    def _build_replay_example_from_texts(self, prompt_text: str, target_text: str, task_idx: int) -> Optional[ReplayExample]:
        if not prompt_text or not target_text:
            return None
        prompt_ids = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=min(self.config.data.max_prompt_length, self.sentinel_sft_max_seq_length),
            add_special_tokens=False,
        )["input_ids"][0]
        remaining = self.sentinel_sft_max_seq_length - int(prompt_ids.size(0))
        if remaining <= 4:
            return None
        target_ids = self.tokenizer(
            target_text,
            return_tensors="pt",
            truncation=True,
            max_length=remaining,
            add_special_tokens=False,
        )["input_ids"][0]
        if target_ids.numel() == 0:
            return None

        input_ids = torch.cat([prompt_ids, target_ids], dim=0).cpu()
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[: prompt_ids.size(0)] = -100
        return ReplayExample(
            input_ids=input_ids,
            attention_mask=attention_mask.cpu(),
            labels=labels.cpu(),
            prompt_text=prompt_text,
            target_text=target_text,
            task_id=task_idx,
        )

    def _collect_replay_examples_from_dataset(self, dataset, task_idx: int) -> list[dict[str, Any]]:
        if self.sentinel_selection_max_examples <= 0:
            return []
        total = len(dataset)
        if total <= 0:
            return []
        indices = list(range(total))
        if self.sentinel_shuffle_candidates:
            self._replay_rng.shuffle(indices)
        indices = indices[: min(total, self.sentinel_selection_max_examples)]

        collected: list[dict[str, Any]] = []
        for idx in indices:
            sample = dataset[idx]
            prompt_text = self._resolve_prompt_text(sample)
            target_text = self._resolve_target_text(sample)
            if not prompt_text or not target_text:
                continue
            example = self._build_replay_example_from_texts(prompt_text, target_text, task_idx)
            if example is None:
                continue
            collected.append(
                {
                    "prompt_text": prompt_text,
                    "target_text": target_text,
                    "task_id": task_idx,
                    "example": example,
                    "embedding": None,
                }
            )
        return collected

    def _ingest_task_replay_candidates(self, task_idx: int) -> None:
        if (
            not self.enable_interference_recall
            or not self.replay_ingest_task_candidates
            or task_idx in self._tasks_with_candidates
        ):
            return
        dataset = self.train_dataset_list[task_idx]
        new_entries = self._collect_replay_examples_from_dataset(dataset, task_idx)
        if not new_entries:
            return
        self._sentinel_candidate_examples.extend(new_entries)
        if self.sentinel_candidate_cap > 0 and len(self._sentinel_candidate_examples) > self.sentinel_candidate_cap:
            self._sentinel_candidate_examples = self._sentinel_candidate_examples[-self.sentinel_candidate_cap :]
        self._tasks_with_candidates.add(task_idx)

    def _seed_replay_candidates_from_jsonl(self, jsonl_path: str, max_examples: int = 0, default_task_idx: int = -1) -> int:
        if not jsonl_path:
            return 0
        if not os.path.isfile(jsonl_path):
            print(f"[Replay] Seed jsonl not found: {jsonl_path}")
            return 0

        limit = int(max_examples) if int(max_examples or 0) > 0 else None
        seeded_entries: list[dict[str, Any]] = []
        try:
            with open(jsonl_path, "r", encoding="utf-8") as handle:
                for line_idx, raw_line in enumerate(handle, start=1):
                    if limit is not None and len(seeded_entries) >= limit:
                        break
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        sample = json.loads(line)
                    except Exception:
                        print(f"[Replay] Skip invalid jsonl line {line_idx} in {jsonl_path}")
                        continue
                    if not isinstance(sample, dict):
                        continue
                    prompt_text = self._resolve_prompt_text(sample)
                    target_text = self._resolve_target_text(sample)
                    if not prompt_text or not target_text:
                        continue
                    seed_task_id = sample.get("task_id", default_task_idx)
                    try:
                        seed_task_id = int(seed_task_id)
                    except Exception:
                        seed_task_id = int(default_task_idx)
                    example = self._build_replay_example_from_texts(prompt_text, target_text, seed_task_id)
                    if example is None:
                        continue
                    seeded_entries.append(
                        {
                            "prompt_text": str(prompt_text),
                            "target_text": str(target_text),
                            "task_id": int(seed_task_id),
                            "example": example,
                            "embedding": None,
                        }
                    )
        except Exception as exc:
            print(f"[Replay] Failed to seed from {jsonl_path}: {str(exc).splitlines()[0]}")
            return 0

        if not seeded_entries:
            print(f"[Replay] Seeded 0 replay candidates from {jsonl_path}")
            return 0

        self._sentinel_candidate_examples.extend(seeded_entries)
        if self.sentinel_candidate_cap > 0 and len(self._sentinel_candidate_examples) > self.sentinel_candidate_cap:
            self._sentinel_candidate_examples = self._sentinel_candidate_examples[-self.sentinel_candidate_cap :]
        self._allow_replay_on_first_task = True
        print(
            f"[Replay] Seeded {len(seeded_entries)} replay candidates from {jsonl_path}. "
            f"Candidate pool size={len(self._sentinel_candidate_examples)}"
        )
        return len(seeded_entries)

    def _seed_replay_candidates_from_jsonl_list(
        self,
        jsonl_paths: list[str],
        max_examples: int = 0,
        max_examples_per_file: int = 0,
    ) -> int:
        paths = [str(path) for path in (jsonl_paths or []) if path]
        if not paths:
            return 0

        total_seeded = 0
        total_budget = int(max_examples or 0)
        per_file_budget = int(max_examples_per_file or 0)
        default_task_idx_base = -len(paths)
        for file_idx, jsonl_path in enumerate(paths):
            if total_budget > 0 and total_seeded >= total_budget:
                break
            remaining = total_budget - total_seeded if total_budget > 0 else 0
            file_budget = remaining
            if per_file_budget > 0:
                file_budget = min(file_budget, per_file_budget) if file_budget > 0 else per_file_budget
            seeded_now = self._seed_replay_candidates_from_jsonl(
                jsonl_path=jsonl_path,
                max_examples=file_budget,
                default_task_idx=default_task_idx_base + file_idx,
            )
            total_seeded += int(seeded_now or 0)
        return total_seeded

    def _maybe_seed_replay_candidates(self) -> None:
        if not self.enable_interference_recall:
            return
        if self._sentinel_candidate_examples:
            if self._allow_replay_on_first_task:
                print(
                    f"[Replay] Reusing {len(self._sentinel_candidate_examples)} preloaded replay candidates from state"
                )
            return
        if not self.seed_jsonl_path_list:
            return

        total_seeded = self._seed_replay_candidates_from_jsonl_list(
            jsonl_paths=self.seed_jsonl_path_list,
            max_examples=int(self.sentinel_selection_max_examples or 0),
            max_examples_per_file=int(self.seed_jsonl_max_examples_per_file or 0),
        )
        if total_seeded > 0:
            # Precompute embeddings so the first task can materialize the replay pool immediately.
            self._ensure_candidate_embeddings(self._sentinel_candidate_examples)

    def _collate_replay_examples(self, examples: list[ReplayExample]) -> dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        input_ids = pad_sequence([ex.input_ids for ex in examples], batch_first=True, padding_value=pad_id)
        attention_mask = pad_sequence([ex.attention_mask for ex in examples], batch_first=True, padding_value=0)
        labels = pad_sequence([ex.labels for ex in examples], batch_first=True, padding_value=-100)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _build_replay_dataproto(self, examples: list[ReplayExample], eval_batch_size: int) -> DataProto:
        batch = self._collate_replay_examples(examples)
        return DataProto.from_dict(tensors=batch, meta_info={"replay_eval_batch_size": max(1, int(eval_batch_size))}, auto_padding=True)

    def _build_embedding_dataproto(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        eval_batch_size: int,
        *,
        return_batch_centroid: bool = False,
    ) -> DataProto:
        return DataProto.from_dict(
            tensors={
                "input_ids": input_ids.cpu(),
                "attention_mask": attention_mask.cpu(),
            },
            meta_info={
                "replay_eval_batch_size": max(1, int(eval_batch_size)),
                "replay_hidden_layer_index": int(self.prototype_hidden_layer_index),
                "return_batch_centroid": bool(return_batch_centroid),
            },
            auto_padding=True,
        )

    def _compute_batch_centroid(self, batch: DataProto) -> Optional[torch.Tensor]:
        if "input_ids" not in batch.batch.keys() or "attention_mask" not in batch.batch.keys():
            return None
        proto = self._build_embedding_dataproto(
            input_ids=batch.batch["input_ids"],
            attention_mask=batch.batch["attention_mask"],
            eval_batch_size=self.sentinel_hidden_state_batch_size,
            return_batch_centroid=True,
        )
        outputs = self.actor_rollout_wg.encode_replay_embeddings(proto)
        if "batch_centroid" not in outputs.batch.keys():
            return None
        centroid = outputs.batch["batch_centroid"]
        if centroid.numel() == 0:
            return None
        return centroid[0].detach().cpu().float()

    def _maybe_update_code_prototype(self, task_idx: int, task_step: int, batch: DataProto, metrics: dict[str, float]) -> None:
        if (
            not self.enable_interference_recall
            or not self._is_code_prototype_task(task_idx)
            or task_step % self.prototype_update_interval != 0
        ):
            return

        centroid = self._compute_batch_centroid(batch)
        if centroid is None or centroid.numel() == 0:
            return

        # Prototype 是“当前任务分布”的 EMA 表征。
        # 后续 replay 打分时会把 sentinel embedding 与这个 prototype 的余弦相似度
        # 作为相关性信号，与 Delta Loss 一起决定哪些历史簇值得优先回放。
        if not self._code_prototype_initialized or self._code_prototype is None or self._code_prototype.numel() == 0:
            self._code_prototype = centroid.clone()
            self._code_prototype_initialized = True
        else:
            decay = float(min(max(self.prototype_ema_decay, 0.0), 1.0))
            self._code_prototype = self._code_prototype * decay + centroid * (1.0 - decay)
        self._code_prototype_source_task = int(task_idx)
        metrics["replay/prototype_available"] = 1.0
        metrics["replay/prototype_norm"] = float(self._code_prototype.norm().item())

    def _ensure_candidate_embeddings(self, entries: list[dict[str, Any]]) -> None:
        # 候选池中的每个 replay example 都需要一个固定隐藏层的 mean-pooled embedding，
        # 后面 sentinel 聚类、prototype 相似度计算都基于这份表征。
        missing = [
            entry
            for entry in entries
            if entry.get("example") is not None
            and (
                entry.get("embedding") is None
                or int(entry.get("embedding_hidden_layer_index", self.prototype_hidden_layer_index))
                != int(self.prototype_hidden_layer_index)
            )
        ]
        if not missing:
            return
        examples = [entry["example"] for entry in missing]
        batch = self._collate_replay_examples(examples)
        proto = self._build_embedding_dataproto(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            eval_batch_size=self.sentinel_hidden_state_batch_size,
        )
        outputs = self.actor_rollout_wg.encode_replay_embeddings(proto)
        embeddings = outputs.batch["embeddings"].cpu()
        if embeddings.size(0) != len(missing):
            raise RuntimeError(
                f"Replay embedding count mismatch: got {embeddings.size(0)} for {len(missing)} examples"
            )
        for entry, embedding in zip(missing, embeddings, strict=True):
            entry["embedding"] = embedding.detach().cpu().float()
            entry["embedding_hidden_layer_index"] = int(self.prototype_hidden_layer_index)

    def _compute_replay_sft_losses(self, examples: list[ReplayExample]) -> list[float]:
        if not examples:
            return []
        proto = self._build_replay_dataproto(examples, self.sft_replay_batch_size)
        outputs = self.actor_rollout_wg.compute_replay_sft_losses(proto)
        losses = outputs.batch["replay_sft_losses"].cpu()
        return [float(x) for x in losses.tolist()]

    @staticmethod
    def _zscore_normalize(values: list[float]) -> list[float]:
        if not values:
            return []
        array = np.asarray(values, dtype=np.float32)
        mean = float(array.mean())
        std = float(array.std())
        if std < 1e-6:
            return [0.0] * len(values)
        normalized = (array - mean) / std
        return [float(x) for x in normalized.tolist()]

    def _run_kmeans(self, embeddings: torch.Tensor, num_clusters: int) -> tuple[list[int], torch.Tensor]:
        data = embeddings.detach().cpu().float()
        num_points = int(data.size(0))
        num_clusters = max(1, min(int(num_clusters), num_points))
        if num_clusters == 1:
            return [0] * num_points, data[:1].clone()

        generator = torch.Generator()
        generator.manual_seed(int(self.config.data.get("seed", 0) or 0))
        init_indices = torch.randperm(num_points, generator=generator)[:num_clusters]
        centroids = data.index_select(0, init_indices).clone()

        for _ in range(self.sentinel_kmeans_iters):
            distances = torch.cdist(data, centroids)
            assignments = torch.argmin(distances, dim=1)
            new_centroids = centroids.clone()
            for cluster_id in range(num_clusters):
                mask = assignments == cluster_id
                if mask.any():
                    new_centroids[cluster_id] = data[mask].mean(dim=0)
            if torch.allclose(new_centroids, centroids):
                centroids = new_centroids
                break
            centroids = new_centroids

        final_assignments = torch.argmin(torch.cdist(data, centroids), dim=1)
        return final_assignments.tolist(), centroids

    def _materialize_sentinel_pool(self) -> None:
        if not self._sentinel_candidate_examples:
            self._sentinel_active_pool = None
            print("[Replay] Sentinel pool skipped: no candidate examples available")
            return
        self._ensure_candidate_embeddings(self._sentinel_candidate_examples)
        embeddings = [entry.get("embedding") for entry in self._sentinel_candidate_examples]
        if not embeddings or any(emb is None for emb in embeddings):
            self._sentinel_active_pool = None
            print("[Replay] Sentinel pool skipped: candidate embeddings are missing")
            return
        embedding_tensor = torch.stack(embeddings, dim=0)
        if self.sentinel_sample_count <= 0:
            self._sentinel_active_pool = None
            print("[Replay] Sentinel pool skipped: sentinel_sample_count <= 0")
            return
        # README 中的 sentinel 机制在这里落地：
        # 先对 replay candidate embedding 做 k-means，再把每个簇里最靠近质心的样本
        # 选成 sentinel，后续只跟踪这些代表点的 loss 变化即可近似监控“哪些历史模式在遗忘”。
        assignments, centroids = self._run_kmeans(
            embedding_tensor, min(self.sentinel_sample_count, embedding_tensor.size(0))
        )
        cluster_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, cluster_id in enumerate(assignments):
            cluster_to_indices[int(cluster_id)].append(idx)

        sentinel_indices: dict[int, int] = {}
        for cluster_id, indices in cluster_to_indices.items():
            cluster_embeddings = embedding_tensor[indices]
            centroid = centroids[cluster_id].to(device=cluster_embeddings.device, dtype=cluster_embeddings.dtype)
            distances = torch.norm(cluster_embeddings - centroid, dim=1)
            sentinel_indices[int(cluster_id)] = indices[int(torch.argmin(distances).item())]

        self._sentinel_active_pool = {
            "candidates": self._sentinel_candidate_examples,
            "cluster_assignments": assignments,
            "cluster_to_indices": dict(cluster_to_indices),
            "sentinel_indices": sentinel_indices,
            "mir_initial_losses": None,
            "mir_initial_task": None,
        }
        print(
            "[Replay] Sentinel pool materialized: "
            f"candidates={len(self._sentinel_candidate_examples)}, "
            f"clusters={len(cluster_to_indices)}, "
            f"sentinels={len(sentinel_indices)}"
        )

    def _initialize_mir_baselines(self, task_idx: int) -> None:
        pool = self._sentinel_active_pool
        if pool is None:
            return
        sentinel_indices = pool.get("sentinel_indices", {})
        if not sentinel_indices:
            return
        examples = []
        cluster_ids = []
        for cluster_id in sorted(sentinel_indices):
            candidate_idx = sentinel_indices[cluster_id]
            example = pool["candidates"][candidate_idx]["example"]
            if example is None:
                continue
            examples.append(example)
            cluster_ids.append(cluster_id)
        if not examples:
            return
        # 训练新任务前，先记录每个 sentinel 的 baseline SFT loss。
        # 之后定期重新评估 current loss，两者差值就是 Delta Loss。
        losses = self._compute_replay_sft_losses(examples)
        pool["mir_initial_losses"] = {cluster_id: float(loss) for cluster_id, loss in zip(cluster_ids, losses, strict=True)}
        pool["mir_initial_task"] = int(task_idx)
        print(
            "[Replay] MIR baselines initialized: "
            f"task={task_idx + 1}, "
            f"sentinel_count={len(cluster_ids)}, "
            f"loss_mean={float(np.mean(losses)) if losses else 0.0:.6f}"
        )

    def _sample_replay_examples(self, selected_clusters: list[int]) -> list[ReplayExample]:
        pool = self._sentinel_active_pool
        if pool is None:
            return []
        samples: list[ReplayExample] = []
        cluster_to_indices = pool.get("cluster_to_indices", {})
        for cluster_id in selected_clusters:
            candidate_indices = list(cluster_to_indices.get(int(cluster_id), []))
            if not candidate_indices:
                continue
            k = min(self.num_replayed_examples_per_cluster, len(candidate_indices))
            if k <= 0:
                continue
            chosen = self._replay_rng.sample(candidate_indices, k) if len(candidate_indices) > k else candidate_indices
            for idx in chosen:
                example = pool["candidates"][idx].get("example")
                if example is not None:
                    samples.append(example)
        return samples

    def _prepare_replay_payload(
        self, task_idx: int, task_step: int, task_total_steps: int
    ) -> tuple[Optional[dict[str, Any]], dict[str, float]]:
        metrics: dict[str, float] = {}
        pool = self._sentinel_active_pool
        if (
            not self.enable_interference_recall
            or (task_idx <= 0 and not self._allow_replay_on_first_task)
            or pool is None
            or not pool.get("sentinel_indices")
            or task_step % self.sentinel_eval_interval != 0
        ):
            return None, metrics

        replay_lambda = self._get_scheduled_replay_loss_lambda(task_step, task_total_steps)
        if replay_lambda <= 0.0:
            return None, metrics

        prototype_available = bool(self._code_prototype_initialized and self._code_prototype is not None)
        metrics["replay/prototype_available"] = 1.0 if prototype_available else 0.0
        if prototype_available:
            metrics["replay/prototype_norm"] = float(self._code_prototype.norm().item())
            prototype = F.normalize(self._code_prototype.view(1, -1), dim=-1)
        else:
            prototype = None

        baseline = pool.get("mir_initial_losses") or {}
        examples = []
        cluster_ids = []
        sentinel_embeddings = []
        for cluster_id in sorted(pool["sentinel_indices"]):
            candidate_idx = pool["sentinel_indices"][cluster_id]
            candidate = pool["candidates"][candidate_idx]
            example = candidate.get("example")
            if example is None or cluster_id not in baseline:
                continue
            examples.append(example)
            cluster_ids.append(cluster_id)
            sentinel_embeddings.append(candidate.get("embedding"))
        if not examples:
            return None, metrics

        # 这里对应 README 里的 replay 打分公式：
        #   score = alpha * z(delta_loss) + beta * z(cosine_to_prototype)
        # Delta Loss 负责找“正在被遗忘”的簇，
        # cosine 负责找“和当前任务更相关、更可能带来正迁移”的簇。
        current_losses = self._compute_replay_sft_losses(examples)
        raw_deltas = []
        raw_cosines = []
        for cluster_id, current_loss, sentinel_embedding in zip(cluster_ids, current_losses, sentinel_embeddings, strict=True):
            delta = float(current_loss) - float(baseline.get(cluster_id, 0.0))
            cosine = 0.0
            if prototype is not None and isinstance(sentinel_embedding, torch.Tensor) and sentinel_embedding.numel() > 0:
                normalized_embedding = F.normalize(sentinel_embedding.view(1, -1).float(), dim=-1)
                cosine = float((normalized_embedding * prototype).sum(dim=-1).item())
            raw_deltas.append(delta)
            raw_cosines.append(cosine)

        normalized_deltas = self._zscore_normalize(raw_deltas)
        normalized_cosines = self._zscore_normalize(raw_cosines)
        scored_clusters = []
        for cluster_id, delta, cosine, delta_norm, cosine_norm in zip(
            cluster_ids,
            raw_deltas,
            raw_cosines,
            normalized_deltas,
            normalized_cosines,
            strict=True,
        ):
            score = float(self.replay_score_alpha * delta_norm + self.replay_score_beta * cosine_norm)
            scored_clusters.append((int(cluster_id), delta, cosine, delta_norm, cosine_norm, score))
        scored_clusters.sort(key=lambda item: item[5], reverse=True)
        top_k = min(self.selected_cluster_num_k, len(scored_clusters))
        selected_clusters = [cluster_id for cluster_id, _, _, _, _, _ in scored_clusters[:top_k]]
        if not selected_clusters:
            return None, metrics

        replay_examples = self._sample_replay_examples(selected_clusters)
        if not replay_examples:
            return None, metrics

        batch = self._collate_replay_examples(replay_examples)
        selected_deltas = [delta for _, delta, _, _, _, _ in scored_clusters[:top_k]]
        selected_cosines = [cosine for _, _, cosine, _, _, _ in scored_clusters[:top_k]]
        selected_delta_norms = [delta_norm for _, _, _, delta_norm, _, _ in scored_clusters[:top_k]]
        selected_cosine_norms = [cosine_norm for _, _, _, _, cosine_norm, _ in scored_clusters[:top_k]]
        selected_scores = [score for _, _, _, _, _, score in scored_clusters[:top_k]]
        metrics.update(
            {
                "replay/lambda": float(replay_lambda),
                "replay/selected_clusters": float(len(selected_clusters)),
                "replay/candidate_clusters": float(len(scored_clusters)),
                "replay/examples": float(len(replay_examples)),
                "replay/mir_delta_sel_mean": float(np.mean(selected_deltas)),
                "replay/mir_delta_sel_min": float(np.min(selected_deltas)),
                "replay/mir_delta_sel_max": float(np.max(selected_deltas)),
                "replay/sentinel_cos_sel_mean": float(np.mean(selected_cosines)),
                "replay/sentinel_cos_sel_min": float(np.min(selected_cosines)),
                "replay/sentinel_cos_sel_max": float(np.max(selected_cosines)),
                "replay/mir_delta_norm_sel_mean": float(np.mean(selected_delta_norms)),
                "replay/mir_delta_norm_sel_min": float(np.min(selected_delta_norms)),
                "replay/mir_delta_norm_sel_max": float(np.max(selected_delta_norms)),
                "replay/sentinel_cos_norm_sel_mean": float(np.mean(selected_cosine_norms)),
                "replay/sentinel_cos_norm_sel_min": float(np.min(selected_cosine_norms)),
                "replay/sentinel_cos_norm_sel_max": float(np.max(selected_cosine_norms)),
                "replay/score_sel_mean": float(np.mean(selected_scores)),
                "replay/score_sel_min": float(np.min(selected_scores)),
                "replay/score_sel_max": float(np.max(selected_scores)),
            }
        )
        payload = {
            "input_ids": batch["input_ids"].cpu(),
            "attention_mask": batch["attention_mask"].cpu(),
            "labels": batch["labels"].cpu(),
            "batch_size": int(self.sft_replay_batch_size),
            "replay_loss_lambda": float(replay_lambda),
        }
        # 这里不直接替换 RL batch，而是把 replay payload 挂到 meta_info。
        # actor update 时会在 RL backward 之后再额外做一次 replay 的 SFT backward，
        # 从而实现 README 里的:
        #   L_total = L_RL + lambda * L_replay
        print(
            "[Replay] Prepared payload: "
            f"task={task_idx + 1}, "
            f"task_step={task_step}, "
            f"lambda={replay_lambda:.6f}, "
            f"selected_clusters={len(selected_clusters)}, "
            f"candidate_clusters={len(scored_clusters)}, "
            f"examples={len(replay_examples)}"
        )
        return payload, metrics

    def _replay_state_path(self, checkpoint_dir: str) -> str:
        return os.path.join(checkpoint_dir, "replay_state.pt")

    def _save_replay_state(self, checkpoint_dir: str) -> None:
        if not self.enable_interference_recall:
            return
        payload = {
            "tasks_with_candidates": sorted(self._tasks_with_candidates),
            "candidate_examples": self._sentinel_candidate_examples,
            "active_pool": self._sentinel_active_pool,
            "allow_replay_on_first_task": bool(self._allow_replay_on_first_task),
            "code_prototype": self._code_prototype,
            "code_prototype_initialized": bool(self._code_prototype_initialized),
            "code_prototype_source_task": self._code_prototype_source_task,
        }
        torch.save(payload, self._replay_state_path(checkpoint_dir))

    def _load_replay_state(self, checkpoint_dir: Optional[str]) -> None:
        if not self.enable_interference_recall or not checkpoint_dir:
            return
        replay_state_path = self._replay_state_path(checkpoint_dir)
        if not os.path.exists(replay_state_path):
            return
        payload = torch.load(replay_state_path, map_location="cpu", weights_only=False)
        self._tasks_with_candidates = set(int(x) for x in payload.get("tasks_with_candidates", []))
        self._sentinel_candidate_examples = payload.get("candidate_examples", []) or []
        self._sentinel_active_pool = payload.get("active_pool", None)
        self._allow_replay_on_first_task = bool(payload.get("allow_replay_on_first_task", False))
        self._code_prototype = payload.get("code_prototype", None)
        self._code_prototype_initialized = bool(payload.get("code_prototype_initialized", self._code_prototype is not None))
        self._code_prototype_source_task = payload.get("code_prototype_source_task", None)
        if self._sentinel_active_pool is not None and isinstance(self._sentinel_active_pool, dict):
            self._sentinel_active_pool["candidates"] = self._sentinel_candidate_examples

    def _resolve_resume_checkpoint_path(self) -> Optional[str]:
        if self.config.trainer.resume_mode == "disable":
            return None
        checkpoint_folder = self.config.trainer.default_local_dir
        if checkpoint_folder is None:
            return None
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
        if self.config.trainer.resume_mode == "auto":
            return find_latest_ckpt_path(checkpoint_folder)
        if self.config.trainer.resume_mode == "resume_path":
            resume_path = self.config.trainer.resume_from_path
            if resume_path is None:
                return None
            if not os.path.isabs(resume_path):
                resume_path = os.path.join(os.getcwd(), resume_path)
            return resume_path
        return None

    def _save_checkpoint(self):
        super()._save_checkpoint()
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        self._save_replay_state(checkpoint_dir)

    def _load_checkpoint(self):
        self._replay_resume_checkpoint_path = self._resolve_resume_checkpoint_path()
        if not hasattr(self, "train_dataloader") and getattr(self, "train_dataloader_list", None):
            self.train_dataloader = self.train_dataloader_list[0]
        super()._load_checkpoint()
        self._load_replay_state(self._replay_resume_checkpoint_path)

    def fit(self):
        """The training loop for continual learning PPO.
        Sequentially trains on multiple tasks while maintaining the same model.
        """

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        self._maybe_seed_replay_candidates()

        # perform validation before training on all tasks
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            print("=" * 80)
            print("Initial validation on all tasks")
            print("=" * 80)
            for task_idx in range(self.config.data.num_tasks):
                print(f"\n--- Validating Task {task_idx + 1}/{self.config.data.num_tasks} ---")
                self.val_dataloader = self.val_dataloader_list[task_idx]
                val_metrics = self._validate()
                if val_metrics:
                    val_metrics = {f"task{task_idx + 1}_init/{k}": v for k, v in val_metrics.items()}
                    logger.log(data=val_metrics, step=self.global_steps)
            
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Continual Learning Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        # ===== Continual Learning Loop: Train on each task sequentially =====
        for task_idx in range(self.config.data.num_tasks):
            print("\n" + "=" * 80)
            print(f"Training on Task {task_idx + 1}/{self.config.data.num_tasks}")
            print("=" * 80)

            task_steps_per_epoch = len(self.train_dataloader_list[task_idx])
            task_total_steps = self.task_step_budgets[task_idx]
            num_task_epochs = max(1, math.ceil(task_total_steps / task_steps_per_epoch))
            task_step = 0
            if (
                self.nsp_enabled
                and self.nsp_projection_mode == "task_end"
                and self.nsp_reset_on_task_start
                and task_idx == self.nsp_anchor_task_id
            ):
                print(f"[NSP] Resetting statistics at anchor task start (task {task_idx + 1})")
                self.actor_rollout_wg.reset_nsp_cache()
            
            # Set current task's dataloaders
            self.train_dataloader = self.train_dataloader_list[task_idx]
            self.val_dataloader = self.val_dataloader_list[task_idx]

            if self.enable_interference_recall and (task_idx > 0 or self._allow_replay_on_first_task):
                # 新任务开始时，把历史 replay candidate 压缩成 sentinel pool，
                # 并为每个 sentinel 记录一份 baseline loss，后续 replay 选择都基于它。
                self._materialize_sentinel_pool()
                self._initialize_mir_baselines(task_idx)

            if self.nsp_enabled and self.nsp_projection_mode == "dynamic" and self.nsp_refresh_first_step:
                nsp_dynamic_init_metrics: dict[str, Any] = {}
                if self._refresh_nsp_from_replay_anchor(task_idx=task_idx, epoch=0, metrics=nsp_dynamic_init_metrics):
                    self.nsp_last_refresh_step = 0
                    nsp_dynamic_init_metrics["nsp/dynamic_refresh"] = 1
                    nsp_dynamic_init_metrics["nsp/dynamic_refresh_task_step"] = 0
                    nsp_dynamic_init_metrics["nsp/dynamic_refresh_reason"] = "task_start"
                    logger.log(data=nsp_dynamic_init_metrics, step=self.global_steps)
            
            # Calculate current_epoch for this task
            current_epoch = 0  # Each task starts from epoch 0

            while task_step < task_total_steps:
                epoch = current_epoch
                print(f"\n--- Task {task_idx + 1}, Epoch {epoch + 1}/{num_task_epochs} ---")
                for batch_dict in self.train_dataloader:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                    metrics = {}
                    timing_raw = {}
                    task_step += 1

                    with marked_timer("start_profile", timing_raw):
                        self._start_profiling(
                            not prev_step_profile and curr_step_profile
                            if self.config.global_profiler.profile_continuous_steps
                            else curr_step_profile
                        )
                    batch: DataProto = DataProto.from_single_dict(batch_dict)
                    batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                    # add uid to batch
                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )

                    gen_batch = self._get_gen_batch(batch)

                    # pass global_steps to trace
                    gen_batch.meta_info["global_steps"] = self.global_steps
                    gen_batch_output = gen_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                    )

                    is_last_step = self.global_steps >= self.total_training_steps
                    with marked_timer("step", timing_raw):
                        # generate a batch
                        with marked_timer("gen", timing_raw, color="red"):
                            if not self.async_rollout_mode:
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                            else:
                                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                            timing_raw.update(gen_batch_output.meta_info["timing"])
                            gen_batch_output.meta_info.pop("timing", None)

                        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                            if self.reward_fn is None:
                                raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                            with marked_timer("gen_max", timing_raw, color="purple"):
                                gen_baseline_batch = deepcopy(gen_batch)
                                gen_baseline_batch.meta_info["do_sample"] = False
                                if not self.async_rollout_mode:
                                    gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                                else:
                                    gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                                batch = batch.union(gen_baseline_output)
                                # compute reward model score on batch
                                rm_scores = None
                                if self.use_rm and "rm_scores" not in batch.batch.keys():
                                    if not self.use_reward_loop:
                                        rm_scores = self.rm_wg.compute_rm_score(batch)
                                    else:
                                        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                        rm_scores = self.reward_loop_manager.compute_rm_score(batch)
                                    batch = batch.union(rm_scores)

                                # Compute or extract reward for REMAX baseline
                                reward_baseline_tensor = self._compute_or_extract_reward(
                                    batch, reward_fn=self.reward_fn, sum_reward=True
                                )

                                keys_to_pop = set(gen_baseline_output.batch.keys())
                                if rm_scores is not None:
                                    keys_to_pop.update(rm_scores.batch.keys())
                                batch.pop(batch_keys=list(keys_to_pop))

                                batch.batch["reward_baselines"] = reward_baseline_tensor

                                del rm_scores, gen_baseline_batch, gen_baseline_output
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)

                        if "response_mask" not in batch.batch.keys():
                            batch.batch["response_mask"] = compute_response_mask(batch)
                        # Balance the number of valid tokens across DP ranks.
                        # NOTE: This usually changes the order of data in the `batch`,
                        # which won't affect the advantage calculation (since it's based on uid),
                        # but might affect the loss calculation (due to the change of mini-batching).
                        if self.config.trainer.balance_batch:
                            self._balance_batch(batch, metrics=metrics)

                        # compute global_valid tokens
                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                        # 当前任务 batch 的隐藏层中心用于更新 prototype，
                        # 让 replay 的“相关性分数”跟着当前任务分布动态变化。
                        self._maybe_update_code_prototype(task_idx=task_idx, task_step=task_step, batch=batch, metrics=metrics)

                        with marked_timer("reward", timing_raw, color="yellow"):
                            # compute reward model score
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                if not self.use_reward_loop:
                                    reward_tensor = self.rm_wg.compute_rm_score(batch)
                                else:
                                    assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                    reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                                batch = batch.union(reward_tensor)

                            # Compute or extract reward for training
                            if self.config.reward_model.launch_reward_fn_async:
                                future_reward = compute_reward_async.remote(
                                    data=batch, config=self.config, tokenizer=self.tokenizer
                                )
                            else:
                                reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                                    batch, reward_fn=self.reward_fn, return_dict=False
                                )

                        # Operating Mode Selection:
                        # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                        # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                        #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                        if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                            from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                            apply_bypass_mode(
                                batch=batch,
                                rollout_corr_config=rollout_corr_config,
                                policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                            )
                        else:  # Recompute old_log_probs
                            with marked_timer("old_log_prob", timing_raw, color="blue"):
                                old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                                entropys = old_log_prob.batch["entropys"]
                                response_masks = batch.batch["response_mask"]
                                actor_config = self.config.actor_rollout_ref.actor
                                entropy_agg = agg_loss(
                                    loss_mat=entropys,
                                    loss_mask=response_masks,
                                    loss_agg_mode=actor_config.loss_agg_mode,
                                    loss_scale_factor=actor_config.loss_scale_factor,
                                )
                                old_log_prob_metrics = {
                                    "actor/entropy": entropy_agg.detach().item(),
                                    "perf/mfu/actor_infer": old_log_prob_mfu,
                                }
                                metrics.update(old_log_prob_metrics)
                                old_log_prob.batch.pop("entropys")
                                batch = batch.union(old_log_prob)
                                if "rollout_log_probs" in batch.batch.keys():
                                    # TODO: we may want to add diff of probs too.
                                    from verl.utils.debug.metrics import calculate_debug_metrics

                                    metrics.update(calculate_debug_metrics(batch))

                        assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                                ref_log_prob = self._compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        # compute values
                        if self.use_critic:
                            with marked_timer("values", timing_raw, color="cyan"):
                                values = self._compute_values(batch)
                                batch = batch.union(values)

                        with marked_timer("adv", timing_raw, color="brown"):
                            # we combine with rule-based rm
                            reward_extra_infos_dict: dict[str, list]
                            if self.config.reward_model.launch_reward_fn_async:
                                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            batch.batch["token_level_scores"] = reward_tensor

                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                            # compute rewards. apply_kl_penalty if available
                            if self.config.algorithm.use_kl_in_reward:
                                batch, kl_metrics = apply_kl_penalty(
                                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                )
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                            # Compute rollout correction: IS weights, rejection sampling, and metrics
                            # Only runs in decoupled mode (computes once per batch using stable π_old)
                            # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                            if (
                                rollout_corr_config is not None
                                and "rollout_log_probs" in batch.batch
                                and not bypass_recomputing_logprobs  # Only in decoupled mode
                            ):
                                from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                                # Compute IS weights, apply rejection sampling, compute metrics
                                batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                                # IS and off-policy metrics already have rollout_corr/ prefix
                                metrics.update(is_metrics)

                            # compute advantages, executed on the driver process
                            norm_adv_by_std_in_grpo = self.config.algorithm.get(
                                "norm_adv_by_std_in_grpo", True
                            )  # GRPO adv normalization factor

                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )

                        # update critic
                        if self.use_critic:
                            with marked_timer("update_critic", timing_raw, color="pink"):
                                critic_output = self._update_critic(batch)
                            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                            metrics.update(critic_output_metrics)

                        # implement critic warmup
                        if self.config.trainer.critic_warmup <= self.global_steps:
                            # update actor
                            with marked_timer("update_actor", timing_raw, color="red"):
                                replay_payload, replay_metrics = self._prepare_replay_payload(
                                    task_idx=task_idx,
                                    task_step=task_step,
                                    task_total_steps=task_total_steps,
                                )
                                if replay_payload is not None:
                                    # actor worker 会读取这份 payload，在 RL backward 结束后
                                    # 再追加一次基于 replay example 的 SFT loss backward。
                                    batch.meta_info["replay_payload"] = replay_payload
                                actor_output = self._update_actor(batch)
                                batch.meta_info.pop("replay_payload", None)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update(actor_output_metrics)
                            metrics.update(replay_metrics)

                        # Log rollout generations if enabled
                        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                        if rollout_data_dir:
                            self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        # Add task_id prefix to validation metrics
                        val_metrics = {f"task{task_idx + 1}/{k}": v for k, v in val_metrics.items()}
                        metrics.update(val_metrics)

                    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    # Check if the conditions for saving a checkpoint are met.
                    # The conditions include a mandatory condition (1) and
                    # one of the following optional conditions (2/3/4):
                    # 1. The save frequency is set to a positive value.
                    # 2. It's the last training step.
                    # 3. The current step number is a multiple of the save frequency.
                    # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                    with marked_timer("stop_profile", timing_raw):
                        next_step_profile = (
                            self.global_steps + 1 in self.config.global_profiler.steps
                            if self.config.global_profiler.steps is not None
                            else False
                        )
                        self._stop_profiling(
                            curr_step_profile and not next_step_profile
                            if self.config.global_profiler.profile_continuous_steps
                            else curr_step_profile
                        )
                        prev_step_profile = curr_step_profile
                        curr_step_profile = next_step_profile

                    steps_duration = timing_raw["step"]
                    self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                    # training metrics - add task_id
                    metrics.update(
                        {
                            "training/global_step": self.global_steps,
                            "training/task_id": task_idx + 1,
                            "training/epoch": epoch,
                        }
                    )
                    # collect metrics
                    metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                    metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                    # TODO: implement actual tflpo and theoretical tflpo
                    n_gpus = self.resource_pool_manager.get_n_gpus()
                    metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                    # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                    # NSP: optionally trigger a refresh event based on global step
                    # 这一步不会修改当前 batch 的 reward/advantage；
                    # 它只负责在 trainer 侧决定“何时重建保护旧知识的投影矩阵”。
                    self._maybe_update_nsp(
                        batch=batch,
                        task_idx=task_idx,
                        epoch=epoch,
                        metrics=metrics,
                        task_step=task_step,
                    )

                    # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                    if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                        self.train_dataloader.sampler.update(batch=batch)

                    # TODO: make a canonical logger that supports various backend
                    logger.log(data=metrics, step=self.global_steps)

                    progress_bar.update(1)
                    self.global_steps += 1

                    if (
                        hasattr(self.config.actor_rollout_ref.actor, "profiler")
                        and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                    ):
                        self.actor_rollout_wg.dump_memory_snapshot(
                            tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                        )

                    if is_last_step:
                        if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                            self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                        pprint(f"Final validation metrics: {last_val_metrics}")
                        progress_bar.close()
                        return

                    # this is experimental and may be changed/removed in the future
                    # in favor of a general-purpose data buffer pool
                    if hasattr(self.train_dataset_list[task_idx], "on_batch_end"):
                        # The dataset may be changed after each training batch
                        self.train_dataset[task_idx].on_batch_end(batch=batch)
                    if task_step >= task_total_steps:
                        break

                current_epoch += 1
            
            if self.enable_interference_recall:
                # 当前任务训练结束后，把它整理成下一轮可用的 replay candidate，
                # 让后续任务可以用 Delta Loss + prototype 相似度来判断是否需要回放。
                self._ingest_task_replay_candidates(task_idx)

            # Task completed - finalize NSP projection at task end (if enabled)
            if self.nsp_enabled and self.nsp_projection_mode == "task_end":
                nsp_task_end_metrics: dict[str, Any] = {}
                last_epoch = max(0, int(current_epoch) - 1)
                self._finalize_nsp_task_end(task_idx=task_idx, epoch=last_epoch, metrics=nsp_task_end_metrics)
                if nsp_task_end_metrics:
                    logger.log(data=nsp_task_end_metrics, step=self.global_steps)

            # Task completed - perform validation on this task
            if self.val_reward_fn is not None:
                print(f"\n--- Final validation on Task {task_idx + 1} after training ---")
                val_metrics = self._validate()
                if val_metrics:
                    val_metrics = {f"task{task_idx + 1}_final/{k}": v for k, v in val_metrics.items()}
                    logger.log(data=val_metrics, step=self.global_steps)

            # Save checkpoint after each task
            if self.config.trainer.save_freq > 0:
                print(f"Saving checkpoint after completing Task {task_idx + 1}")
                self._save_checkpoint()
        
        # All tasks completed
        progress_bar.close()
        print("\n" + "=" * 80)
        print("Continual Learning Training Completed!")
        print("=" * 80)
