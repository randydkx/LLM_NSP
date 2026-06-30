import ast
import json
import math
import os
import re
import resource
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List


# Enable/disable think-length gating. When enabled, reward is scaled (only on success)
# by a function of the <think>...</think> length.
use_think_length_gating = bool(int(os.environ.get("USE_THINK_LENGTH_GATING", "0")))

# Hyperparameters for think-length gating (tuned to be small by default).
_THINK_LEN_ALPHA = float(os.environ.get("THINK_LENGTH_ALPHA", "0.10"))
_THINK_LEN_REF_TOKENS = int(os.environ.get("THINK_LENGTH_REF_TOKENS", "128"))
_THINK_LEN_MAX_SCALE = float(os.environ.get("THINK_LENGTH_MAX_SCALE", "1.20"))

# Hyperparameters for completion-length suitability reward.
_LENGTH_L_MAX = int(os.environ.get("COMPLETION_LENGTH_L_MAX", "7168"))
_LENGTH_L_CACHE = int(os.environ.get("COMPLETION_LENGTH_L_CACHE", "2048"))


# ======================================================
# 1. Utility: Extract Python code blocks
# ======================================================
def extract_code(completion: str) -> str:
    pattern = re.compile(rf"```python\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    return (matches[-1].strip() if matches else "").strip()


# ======================================================
# 2. Atomic execution worker
# ======================================================
def _execute_code_with_tests(
    code: str,
    examples: List[Dict],
    time_limit: float,
    memory_limit: float
) -> float:
    """Executes code against test cases; partial credit per passed case."""

    if not code.strip() or not examples:
        return -1.0

    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py', encoding='utf-8') as f:
            f.write(code)
            file_path = f.name
    except Exception:
        return -1.0

    def set_limits():
        try:
            mem_bytes = int(memory_limit * 1024 * 1024)
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            resource.setrlimit(resource.RLIMIT_CPU, (int(time_limit) + 1, int(time_limit) + 1))
        except Exception:
            pass

    total_cases = len(examples)
    passed_cases = 0

    try:
        for case in examples:
            proc = subprocess.run(
                ["python", file_path],
                input=case.get("input", ""),
                text=True,
                capture_output=True,
                timeout=time_limit,
                preexec_fn=set_limits if os.name != "nt" else None
            )

            # 若执行失败 -> 本测试直接视为失败
            if proc.returncode != 0:
                continue

            # stdout 比对
            if proc.stdout.strip() == case.get("output", "").strip():
                passed_cases += 1

        # 按比例奖励，通过率即 reward
        if total_cases == 0:
            return -1.0

        reward_ratio = passed_cases / total_cases

        # 若完全无执行（全部异常）则记为执行错误
        if passed_cases == 0 and reward_ratio == 0.0:
            # 检查是否可能全执行失败
            return 0.0 if total_cases > 0 else -1.0

        return round(reward_ratio, 4)

    except subprocess.TimeoutExpired:
        return -1.0
    except Exception:
        return -1.0
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# ======================================================
# 3. Core parallel engine
# ======================================================
def parallel_eval_engine(items: List[Dict], max_workers: int) -> List[float]:
    results = [-1.0] * len(items)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                _execute_code_with_tests,
                item["code"],
                item["examples"],
                item["time_limit"],
                item["memory_limit"]
            ): idx
            for idx, item in enumerate(items)
        }
        for f in as_completed(future_map):
            idx = future_map[f]
            try:
                results[idx] = f.result(timeout=15)
            except Exception:
                results[idx] = -1.0
    return results

def sequential_eval_engine(items):
    results = [-1.0] * len(items)
    for idx, item in enumerate(items):
        result = _execute_code_with_tests(item["code"],item["examples"],item["time_limit"],item["memory_limit"])
        results[idx] = result
    return results



# ======================================================
# 4. accuracy_reward: DeepSeek‑R1 样式的底层 rule-based 评估
# ======================================================
def accuracy_reward(
    completions: List[str],
    examples: List[List[Dict]],
    time_limit: List[float],
    memory_limit: List[float],
    **kwargs
) -> List[float]:
    items = [
        {
            "code": extract_code(completions[i]),
            "examples": examples[i],
            "time_limit": time_limit[i],
            "memory_limit": memory_limit[i]
        }
        for i in range(len(completions))
    ]
    scores = sequential_eval_engine(items)

    return scores


# ======================================================
# 5. High-level code_reward (multi-signal aggregation)
# ======================================================
def _syntax_reward(code: str) -> float:
    """Reward valid syntax."""
    try:
        ast.parse(code)
        return 1.0
    except SyntaxError:
        return 0.0
    except Exception:
        return 0.5


def _format_reward(completion: str) -> float:
    """Reward if code is properly formatted in ```python``` block."""
    return 1.0 if "```python" in completion else 0.5 if "```" in completion else 0.0


def _safety_reward(code: str) -> float:
    """Penalize unsafe imports or dangerous calls."""
    blacklist = ["os.system", "subprocess.Popen", "eval(", "exec(", "open(", "socket"]
    return 0.0 if any(b in code for b in blacklist) else 1.0


def _readability_reward(code: str) -> float:
    """Lightweight heuristic for readability (length, comments, blank lines)."""
    lines = [ln.strip() for ln in code.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    comment_ratio = sum(1 for l in lines if l.startswith("#")) / len(lines)
    avg_len = sum(len(l) for l in lines) / len(lines)
    if comment_ratio >= 0.2 and avg_len < 100:
        return 1.0
    elif comment_ratio >= 0.05:
        return 0.7
    return 0.4


def _thinking_reward(completion: str) -> float:
    """Reward if completion contains complete think tags.
    
    Checks for the presence of both <think> and </think> tags in the completion.
    Returns:
        1.0 if both opening and closing think tags are present (complete)
        -1.0 otherwise (incomplete or missing)
    """
    has_opening_tag = "<think>" in completion
    has_closing_tag = "</think>" in completion
    
    if has_opening_tag and has_closing_tag:
        return 1.0
    return -1.0


def _extract_think_text(completion: str) -> str:
    """Extract content inside <think>...</think>. Returns empty string if missing."""
    if not completion:
        return ""
    m = re.search(r"<think>(.*?)</think>", completion, flags=re.DOTALL)
    return (m.group(1) if m else "")


def _count_think_tokens(think_text: str) -> int:
    """Approximate token count for think content without a tokenizer."""
    if not think_text:
        return 0
    return len(re.findall(r"\S+", think_text))


def _think_length_scale(
    think_len_tokens: int,
    *,
    alpha: float = _THINK_LEN_ALPHA,
    ref_tokens: int = _THINK_LEN_REF_TOKENS,
    max_scale: float = _THINK_LEN_MAX_SCALE,
) -> float:
    """Compute a multiplicative scale >=1 based on think length.

    Uses a smooth log curve and caps the maximum multiplier.
    """
    if think_len_tokens <= 0 or alpha <= 0.0:
        return 1.0
    ref_tokens = int(ref_tokens or 0)
    if ref_tokens <= 0:
        return 1.0
    # Normalize into [0, 1] approximately when think_len ~= ref_tokens.
    denom = math.log1p(float(ref_tokens))
    if denom <= 0.0:
        return 1.0
    frac = math.log1p(float(think_len_tokens)) / denom
    scale = 1.0 + float(alpha) * float(max(0.0, frac))
    if max_scale is not None:
        try:
            scale = min(float(max_scale), scale)
        except Exception:
            pass
    return max(1.0, scale)


def _repetition_penalty_text(completion: str) -> float:
    """Penalty for excessive repetition in the completion text.

    Adapted from `reward/standard/math_reward.py` repetition_penalty.
    """
    tokens = re.findall(r"\S+", completion or "")
    if not tokens or len(tokens) < 20:
        return 0.0

    total_tokens = len(tokens)
    ngram_sizes = [4, 8]
    max_repeat_ratio = 0.0

    for n in ngram_sizes:
        if total_tokens < n:
            continue
        ngrams = [tuple(tokens[i:i + n]) for i in range(total_tokens - n + 1)]
        counts = Counter(ngrams)
        most_common_count = counts.most_common(1)[0][1]
        repeat_ratio = (most_common_count * n) / total_tokens
        max_repeat_ratio = max(max_repeat_ratio, repeat_ratio)

    if max_repeat_ratio <= 0.15:
        return 0.0
    elif max_repeat_ratio <= 0.30:
        return -1.0 * ((max_repeat_ratio - 0.15) / 0.15 * 1.0)
    elif max_repeat_ratio <= 0.60:
        return -1.0 - ((max_repeat_ratio - 0.30) / 0.30 * 4.0)
    else:
        penalty = -5.0 - ((max_repeat_ratio - 0.60) / 0.40 * 5.0)
        return max(penalty, -10.0)


def suitable_completion_length_reward(
    completion_ids,
    *,
    l_max: int = _LENGTH_L_MAX,
    l_cache: int = _LENGTH_L_CACHE,
) -> float:
    """Length suitability reward using completion_ids length.

    Piecewise definition (see provided formula):
      - 0,                         if |y| <= L_max - L_cache
      - (L_max - L_cache - |y|)/L_cache, if L_max - L_cache < |y| <= L_max
      - -1,                        if |y| > L_max
    If completion_ids is None, returns 0.
    """
    if completion_ids is None:
        return 0.0
    try:
        length = len(completion_ids)
    except Exception:
        return 0.0

    if l_cache <= 0 or l_max <= 0:
        return 0.0

    threshold = l_max - l_cache
    if length <= threshold:
        return 0.0
    if length <= l_max:
        return (threshold - float(length)) / float(l_cache)
    return -1.0


def code_reward(
    completions: List[str],
    examples: List[List[Dict]],
    time_limit: List[float],
    memory_limit: List[float],
    prompts: List[str] = None,
    completion_ids_list = None,
    **kwargs,
) -> List[float]:
    """
    High-level aggregated reward based on DeepSeek‑R1 code task design.
    Combines multiple dimensions:
      - execution accuracy          (from accuracy_reward)
      - syntax correctness
      - formatting quality
      - safety
      - readability
    """

    # base execution correctness
    exec_scores = accuracy_reward(
        completions,
        examples,
        time_limit,
        memory_limit,
        _skip_logging=True,
        **kwargs,
    )

    all_rewards: List[float] = []
    syntax_list: List[float] = []
    format_list: List[float] = []
    safety_list: List[float] = []
    readability_list: List[float] = []
    thinking_list: List[float] = []
    think_len_tokens_list: List[int] = []
    think_scale_list: List[float] = []
    length_reward_list: List[float] = []

    for i, comp in enumerate(completions):
        thinking = _thinking_reward(comp)
        thinking_list.append(thinking)

        think_text = _extract_think_text(comp)
        think_len_tokens = _count_think_tokens(think_text)
        think_len_tokens_list.append(int(think_len_tokens))

        rep_penalty = _repetition_penalty_text(comp)

        completion_ids = None
        if completion_ids_list is not None and i < len(completion_ids_list):
            completion_ids = completion_ids_list[i]
        length_reward = suitable_completion_length_reward(
            completion_ids,
            l_max=int(kwargs.get("completion_length_l_max", _LENGTH_L_MAX)),
            l_cache=int(kwargs.get("completion_length_l_cache", _LENGTH_L_CACHE)),
        )
        length_reward_list.append(float(length_reward))

        # Hard gate: if the model doesn't provide a complete <think>...</think>,
        # the reward is forced to -1 regardless of execution/other signals.
        if thinking < 1.0:
            syntax_list.append(0.0)
            format_list.append(0.0)
            safety_list.append(0.0)
            readability_list.append(0.0)
            combined = -1.0 + rep_penalty
            combined = max(-1.0, min(1.0, combined))
            all_rewards.append(round(combined, 4))
            continue

        code = extract_code(comp)
        syntax = _syntax_reward(code)
        fmt = _format_reward(comp)
        safety = _safety_reward(code)
        readb = _readability_reward(code)
        exec_val = exec_scores[i]

        syntax_list.append(syntax)
        format_list.append(fmt)
        safety_list.append(safety)
        readability_list.append(readb)

        # If the model didn't produce a runnable code block at all, keep a strong negative signal.
        if exec_val == -1.0 and not code.strip():
            all_rewards.append(-1.0)
            continue

        # --------------------------------------------------------------
        # IMPORTANT: Avoid a short-output loophole.
        #
        # - Execution errors/timeouts must stay negative (do NOT clamp to 0).
        # - When execution doesn't pass any tests, auxiliary bonuses are gated
        #   so the model can't get a stable positive reward from just emitting
        #   a tiny well-formatted code block.
        # --------------------------------------------------------------
        # Strict failure handling:
        # - exec_val == -1.0 (timeout/runtime error): keep strong negative.
        # - exec_val == 0.0 (runs but fails all tests): also negative.
        if exec_val <= 0.0:
            
            # 直接不给任何机会，只要一个样本都没通过就给-1.0
            combined = -1.0
            
            # 下面这个是程序可以执行但是一个样例都没有通过，这种情况会酌情给分
            # combined = -0.25

            # # Make empty/trivial stubs even worse.
            # nonempty_lines = [ln for ln in code.splitlines() if ln.strip()]
            # if len(nonempty_lines) < 3 or len(code.strip()) < 40:
            #     combined -= 0.25

            # # Also penalize missing python fence on failures (formatting matters, but never enough to be positive).
            # if "```python" not in comp:
            #     combined -= 0.15

        else:
            # When at least one test passes, give small shaping rewards.
            bonus = 0.04 * syntax + 0.04 * fmt + 0.015 * safety + 0.005 * readb
            combined = 0.95 * float(exec_val) + bonus

            # Optional: prefer longer thinking among similarly good solutions by scaling the
            # *success* reward with a bounded, smooth function of think length.
            gating_on = bool(kwargs.get("use_think_length_gating", use_think_length_gating))
            if gating_on:
                scale = _think_length_scale(
                    think_len_tokens,
                    alpha=float(kwargs.get("think_length_alpha", _THINK_LEN_ALPHA)),
                    ref_tokens=int(kwargs.get("think_length_ref_tokens", _THINK_LEN_REF_TOKENS)),
                    max_scale=float(kwargs.get("think_length_max_scale", _THINK_LEN_MAX_SCALE)),
                )
                combined = combined * scale
                think_scale_list.append(float(scale))
            else:
                think_scale_list.append(1.0)

        # Keep lists aligned even when failure branches skip gating.
        if len(think_scale_list) < len(all_rewards) + 1:
            think_scale_list.append(1.0)

        combined += rep_penalty

        # Add length suitability reward.
        combined += length_reward

        # Keep reward in a sane range.
        combined = max(-1.0, min(1.0, combined))
        all_rewards.append(round(combined, 4))

    return all_rewards
