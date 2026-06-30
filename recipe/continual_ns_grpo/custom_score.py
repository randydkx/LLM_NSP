import json
import os
import re
from functools import lru_cache
from typing import Any, Optional

from verl.utils.import_utils import load_extern_object


_MATH_DATA_SOURCES = {
    "openai/gsm8k",
    "aime25",
    "aime",
    "open-r1/dapo-math-17k-processed",
    "lighteval/math",
    "digitallearninggmbh/math-lighteval",
    "huggingfaceh4/math-500",
    "math_dapo",
    "dapo_math",
    "math",
    "math_dapo_reasoning",
    "numina_aops_forum",
    "numina_synthetic_math",
    "numina_amc_aime",
    "numina_synthetic_amc",
    "numina_cn_k12",
    "numina_olympiads",
}

_CODE_DATA_SOURCES = {
    "codecontests",
    "apps",
    "codeforces",
    "taco",
    "code_train"
}


def _normalize_data_source(data_source: Any) -> str:
    return str(data_source).strip().lower()


def _is_math_data_source(data_source: Any) -> bool:
    normalized = _normalize_data_source(data_source)
    return normalized in _MATH_DATA_SOURCES or normalized.startswith("aime")


def _is_code_data_source(data_source: Any) -> bool:
    return _normalize_data_source(data_source) in _CODE_DATA_SOURCES


def _think_format_reward(solution_str: str, complete_reward: float, missing_reward: float) -> float:
    think_l = solution_str.find("<think>")
    think_r = solution_str.find("</think>")
    is_complete = think_l >= 0 and think_r > think_l
    return complete_reward if is_complete else missing_reward


@lru_cache(maxsize=1)
def _load_accuracy_reward_fn():
    this_dir = os.path.dirname(__file__)
    raw_accuracy_path = os.path.abspath(
        os.path.join(this_dir, "..", "..", "examples", "grpo_trainer", "math_reward_accuracy_raw.py")
    )
    fn = load_extern_object(module_path=raw_accuracy_path, object_name="accuracy_reward")
    if not callable(fn):
        raise TypeError(f"accuracy_reward is not callable in {raw_accuracy_path}")
    return fn

def extract_last_boxed(text: str) -> Optional[str]:
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[-1].strip() if matches else None

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _maybe_json_load(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _normalize_code_examples(value: Any) -> list[dict[str, str]]:
    payload = _maybe_json_load(value)

    if isinstance(payload, dict):
        if "examples" in payload:
            payload = payload["examples"]
        elif "test_cases" in payload:
            payload = payload["test_cases"]
        elif "inputs" in payload and "outputs" in payload:
            inputs = list(payload.get("inputs") or [])
            outputs = list(payload.get("outputs") or [])
            return [{"input": str(inp), "output": str(out)} for inp, out in zip(inputs, outputs)]

    if isinstance(payload, list):
        normalized = []
        for item in payload:
            item = _maybe_json_load(item)
            if not isinstance(item, dict):
                continue
            if "input" in item and "output" in item:
                normalized.append({"input": str(item["input"]), "output": str(item["output"])})
            elif "inputs" in item and "outputs" in item:
                inputs = list(item.get("inputs") or [])
                outputs = list(item.get("outputs") or [])
                normalized.extend({"input": str(inp), "output": str(out)} for inp, out in zip(inputs, outputs))
        return normalized

    return []


def _extract_code_reward_inputs(
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]],
    kwargs: dict[str, Any],
) -> tuple[list[dict[str, str]], float, float]:
    payload = _maybe_json_load(ground_truth)
    extra_info = extra_info or {}

    examples = _normalize_code_examples(payload)
    if not examples:
        examples = _normalize_code_examples(extra_info.get("examples"))
    if not examples:
        examples = _normalize_code_examples(extra_info.get("test_cases"))

    time_limit = kwargs.get("time_limit")
    if time_limit is None and isinstance(payload, dict):
        time_limit = payload.get("time_limit")
    if time_limit is None:
        time_limit = extra_info.get("time_limit")

    memory_limit = kwargs.get("memory_limit")
    if memory_limit is None and isinstance(payload, dict):
        memory_limit = payload.get("memory_limit")
    if memory_limit is None:
        memory_limit = extra_info.get("memory_limit")

    return examples, _safe_float(time_limit, default=5.0), _safe_float(memory_limit, default=256.0)


def _compute_math_score(
    solution_str: str,
    ground_truth: str,
    accuracy_weight: float = 1.0,
    format_weight: float = 0.1,
    format_complete_reward: float = 1.0,
    format_missing_reward: float = -1.0,
) -> dict[str, float]:
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


def _compute_code_score(
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> dict[str, float]:
    from recipe.continual_ns_grpo.code_reward import code_reward
    examples, time_limit, memory_limit = _extract_code_reward_inputs(
        ground_truth=ground_truth,
        extra_info=extra_info,
        kwargs=kwargs,
    )
    score = code_reward(
        completions=[solution_str],
        examples=[examples],
        time_limit=[time_limit],
        memory_limit=[memory_limit],
        **kwargs,
    )[0]
    score = _safe_float(score, default=-1.0)
    return {
        "score": score,
        "is_correct": 1.0 if score >= 1.0 else 0.0,
    }


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
    if _is_math_data_source(data_source):
        return _compute_math_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            accuracy_weight=accuracy_weight,
            format_weight=format_weight,
            format_complete_reward=format_complete_reward,
            format_missing_reward=format_missing_reward,
        )

    if _is_code_data_source(data_source):
        return _compute_code_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **kwargs,
        )

    raise NotImplementedError(f"Unsupported data_source for continual_ns_grpo custom score: {data_source!r}")
