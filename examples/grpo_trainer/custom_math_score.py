import os
from functools import lru_cache
from typing import Any, Optional

from verl.utils.import_utils import load_extern_object

def _think_format_reward(solution_str: str, complete_reward: float, missing_reward: float) -> float:
    think_l = solution_str.find("<think>")
    think_r = solution_str.find("</think>")
    is_complete = think_l >= 0 and think_r > think_l
    return complete_reward if is_complete else missing_reward


@lru_cache(maxsize=1)
def _load_accuracy_reward_fn():
    this_dir = os.path.dirname(__file__)
    raw_accuracy_path = os.path.join(this_dir, "math_reward_accuracy_raw.py")
    fn = load_extern_object(module_path=raw_accuracy_path, object_name="accuracy_reward")
    if not callable(fn):
        raise TypeError(f"accuracy_reward is not callable in {raw_accuracy_path}")
    return fn


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict[str, Any]] = None,
    accuracy_weight: float = 1.0,
    format_weight: float = 0.1,
    format_complete_reward: float = 1.0,
    format_missing_reward: float = -1.0,
    **kwargs: Any,
) -> dict[str, float]:
    del data_source, extra_info, kwargs

    acc_fn = _load_accuracy_reward_fn()
    acc_reward = _safe_float(acc_fn(solution_str, ground_truth), default=0.0)
    is_correct = 1.0 if acc_reward == 1.0 else 0.0

    fmt_reward = _think_format_reward(
        solution_str=solution_str,
        complete_reward=float(format_complete_reward),
        missing_reward=float(format_missing_reward),
    )

    total = float(accuracy_weight) * acc_reward + float(format_weight) * fmt_reward

    return {
        "score": total,
        "acc_reward": acc_reward,
        "format_reward": fmt_reward,
        "is_correct": is_correct,
    }
