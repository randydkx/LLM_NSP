import os
from collections import defaultdict
from typing import Any, Union

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager.abstract import AbstractRewardManager


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _is_aime25_data_source(data_source: Any) -> bool:
    return str(data_source).strip().lower() == "aime25"


def _binary_acc_score(acc_reward: float) -> float:
    return 1.0 if abs(acc_reward - 1.0) < 1e-8 else 0.0


class CustomMathRewardManager(AbstractRewardManager):
    """Custom reward manager for math GRPO:
    - base reward from custom compute_score (accuracy + think-format)
    - DAPO-style overlong length penalty
    """

    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        max_resp_len: Any = None,
        overlong_buffer_cfg: Any = None,
        length_reward_cfg: Any = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.max_resp_len = int(max_resp_len) if max_resp_len is not None else None
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.length_reward_cfg = length_reward_cfg
        self._effective_overlong_buffer_cfg, self._overlong_cfg_source = self._normalize_overlong_cfg(
            overlong_buffer_cfg=overlong_buffer_cfg,
            length_reward_cfg=length_reward_cfg,
        )
        self._validate_overlong_cfg(self._effective_overlong_buffer_cfg)
        self._debug_runtime_printed = False

    def _normalize_overlong_cfg(self, overlong_buffer_cfg: Any, length_reward_cfg: Any) -> tuple[dict[str, Any], str]:
        if overlong_buffer_cfg is not None:
            raw_cfg = overlong_buffer_cfg
            source = "overlong_buffer_cfg"
        else:
            legacy_len = _cfg_get(length_reward_cfg, "len", None)
            legacy_penalty = _cfg_get(length_reward_cfg, "penalty_factor", None)
            if legacy_len is not None and legacy_penalty is not None:
                raw_cfg = length_reward_cfg
                source = "length_reward_cfg_as_overlong_buffer_cfg"
            else:
                raw_cfg = {"enable": False, "len": 0, "penalty_factor": 0.0, "log": False}
                source = "disabled_legacy_length_reward_cfg_without_len"

        normalized = {
            "enable": bool(_cfg_get(raw_cfg, "enable", False)),
            "len": int(_cfg_get(raw_cfg, "len", 0)),
            "penalty_factor": float(_cfg_get(raw_cfg, "penalty_factor", 0.0)),
            "log": bool(_cfg_get(raw_cfg, "log", False)),
        }
        return normalized, source

    def _validate_overlong_cfg(self, cfg: dict[str, Any]) -> None:
        if not cfg["enable"]:
            return

        overlong_buffer_len = int(cfg["len"])
        assert self.max_resp_len is not None, (
            f"max_resp_len must be provided if overlong penalty is enabled, but got {self.max_resp_len}"
        )
        assert overlong_buffer_len > 0, (
            f"overlong_buffer_cfg.len must be positive if overlong penalty is enabled, but got {overlong_buffer_len}"
        )
        assert self.max_resp_len >= overlong_buffer_len, (
            f"max_resp_len ({self.max_resp_len}) must be larger than or equal to overlong_buffer_cfg.len "
            f"({overlong_buffer_len})"
        )

    def __call__(self, data: DataProto, return_dict: bool = False) -> Union[torch.Tensor, dict[str, Any]]:
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        if (not self._debug_runtime_printed) and self.num_examine > 0:
            print("[debug_reward_manager_file]", os.path.abspath(__file__))
            print("[debug_reward_fn_key]", self.reward_fn_key)
            print("[debug_length_reward_cfg]", self.length_reward_cfg)
            print("[debug_overlong_buffer_cfg_raw]", self.overlong_buffer_cfg)
            print("[debug_overlong_cfg_source]", self._overlong_cfg_source)
            print("[debug_effective_overlong_buffer_cfg]", self._effective_overlong_buffer_cfg)
            print("[debug_max_resp_len]", self.max_resp_len)
            self._debug_runtime_printed = True

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}
        records = []

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum().item())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length_raw = int(data_item.batch["attention_mask"][prompt_length:].sum().item())
            valid_response_length = max(valid_response_length_raw, 1)
            valid_response_ids = response_ids[:valid_response_length_raw]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if eos_token and response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            uid = str(data_item.non_tensor_batch.get("uid", i))
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["rollout_reward_scores"] = rollout_reward_scores

            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(result, dict):
                base_score = _to_float(result.get("score", 0.0), default=0.0)
                acc_reward = _to_float(result.get("acc_reward", base_score), default=base_score)
                is_correct = int(_to_float(result.get("is_correct", _binary_acc_score(acc_reward)), default=0.0))
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                base_score = _to_float(result, default=0.0)
                acc_reward = base_score
                is_correct = int(_binary_acc_score(acc_reward))
                reward_extra_info["score"].append(base_score)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            records.append(
                {
                    "i": i,
                    "uid": uid,
                    "data_source": data_source,
                    "prompt_str": prompt_str,
                    "response_str": response_str,
                    "ground_truth": ground_truth,
                    "response_length": valid_response_length_raw,
                    "terminal_idx": valid_response_length - 1,
                    "base_score": base_score,
                    "acc_reward": acc_reward,
                    "is_correct": is_correct,
                    "use_accuracy_only": _is_aime25_data_source(data_source),
                }
            )

        overlong_cfg = self._effective_overlong_buffer_cfg
        overlong_enabled = bool(overlong_cfg["enable"])
        overlong_buffer_len = int(overlong_cfg["len"])
        overlong_penalty_factor = float(overlong_cfg["penalty_factor"])
        overlong_log = bool(overlong_cfg["log"])

        for ridx, rec in enumerate(records):
            overlong_expected_len = 0.0
            overlong_exceed_len = 0.0
            overlong_reward = 0.0
            is_overlong = False

            if rec["use_accuracy_only"]:
                final_score = float(rec["is_correct"])
            else:
                final_score = rec["base_score"]
                if overlong_enabled:
                    overlong_expected_len = self.max_resp_len - overlong_buffer_len
                    overlong_exceed_len = rec["response_length"] - overlong_expected_len
                    overlong_reward = min(
                        -overlong_exceed_len / float(overlong_buffer_len) * overlong_penalty_factor,
                        0.0,
                    )
                    final_score += overlong_reward
                    is_overlong = overlong_reward < 0.0

            reward_tensor[rec["i"], rec["terminal_idx"]] = final_score
            reward_extra_info["response_length"].append(rec["response_length"])
            reward_extra_info["overlong_expected_len"].append(overlong_expected_len)
            reward_extra_info["overlong_exceed_len"].append(overlong_exceed_len)
            reward_extra_info["overlong_penalty_factor"].append(overlong_penalty_factor if overlong_enabled else 0.0)
            reward_extra_info["final_score"].append(final_score)
            reward_extra_info["use_accuracy_only"].append(int(rec["use_accuracy_only"]))
            if overlong_log:
                reward_extra_info["overlong_reward"].append(overlong_reward)
                reward_extra_info["overlong"].append(is_overlong)

            data_source = rec["data_source"]
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", rec["prompt_str"])
                print("[response]", rec["response_str"])
                print("[ground_truth]", rec["ground_truth"])
                print("[data_source_raw]", repr(rec["data_source"]))
                print("[use_accuracy_only]", rec["use_accuracy_only"])
                print("[acc_reward]", rec["acc_reward"])
                print("[is_correct]", rec["is_correct"])
                print("[base_score]", rec["base_score"])
                print("[overlong_expected_len]", overlong_expected_len)
                print("[overlong_exceed_len]", overlong_exceed_len)
                print("[overlong_penalty_factor]", overlong_penalty_factor if overlong_enabled else 0.0)
                print("[overlong_reward]", overlong_reward)
                print("[final_score]", final_score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor
