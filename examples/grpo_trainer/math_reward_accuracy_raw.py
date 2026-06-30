import re
import math
import os
import json
import time
import datetime
import traceback
import platform
import sys
import contextvars
from typing import List, Optional
from latex2sympy2_extended import NormalizationConfig, latex2sympy, normalize_latex
from math_verify import LatexExtractionConfig, parse, verify
from sympy import sympify, Eq, factorial2, Integer
from sympy.core.containers import Tuple as SympyTuple
from collections import Counter
import warnings
import numpy as np
warnings.filterwarnings("ignore")


log_results = True

# Log exceptions into the same per-run directory as reward records.
# This is best-effort and must never crash training.
log_exceptions = bool(int(os.environ.get("MATH_REWARD_LOG_EXCEPTIONS", "1")))

# Internal helper functions historically swallowed many exceptions (to keep parsing robust).
# If you want full observability, keep this enabled.
log_internal_exceptions = bool(int(os.environ.get("MATH_REWARD_LOG_INTERNAL_EXCEPTIONS", "1")))

# math_verify.parse uses signal.alarm() when parsing_timeout is set, which breaks
# in threaded workers. Default to None for thread-safe behavior.
_MATH_VERIFY_PARSING_TIMEOUT = None
_parse_timeout_env = os.environ.get("MATH_VERIFY_PARSING_TIMEOUT")
if _parse_timeout_env is not None:
    _timeout_text = _parse_timeout_env.strip().lower()
    if _timeout_text in {"", "none", "null"}:
        _MATH_VERIFY_PARSING_TIMEOUT = None
    else:
        try:
            _MATH_VERIFY_PARSING_TIMEOUT = float(_parse_timeout_env)
        except Exception:
            _MATH_VERIFY_PARSING_TIMEOUT = None

# Buffer for internal exceptions swallowed by helper functions.
# We only flush (write to exceptions.json) from the outer `math_reward` loop.
_INTERNAL_EXCEPTION_BUFFER = contextvars.ContextVar("_MATH_REWARD_INTERNAL_EXC_BUFFER", default=None)


def _start_internal_exception_buffer() -> None:
    if not log_internal_exceptions:
        return
    _INTERNAL_EXCEPTION_BUFFER.set([])


def _record_internal_exception(where: str, exc: BaseException, **context) -> None:
    """Record an exception without performing any file I/O."""
    if not log_internal_exceptions:
        return
    try:
        buf = _INTERNAL_EXCEPTION_BUFFER.get()
        if buf is None:
            return
        max_chars = int(os.environ.get("MATH_REWARD_EXCEPTION_MAX_CHARS", "2000"))
        buf.append(
            {
                "where": where,
                "exc_type": type(exc).__name__,
                "exc_msg": _safe_truncate(str(exc), max_chars),
                "traceback": _safe_truncate(traceback.format_exc(), max_chars * 3),
                "context": {k: _safe_truncate(v, max_chars) for k, v in context.items()},
            }
        )
    except Exception:
        # Never break training due to exception recording.
        pass


def _flush_internal_exceptions(*, sample_idx: int, solution: str, completion: str) -> None:
    """Flush buffered internal exceptions to exceptions.json (best-effort)."""
    if not (log_exceptions and log_internal_exceptions):
        return
    try:
        buf = _INTERNAL_EXCEPTION_BUFFER.get()
        if not buf:
            return
        for item in buf:
            _log_exception(
                "math_reward/internal_exception",
                Exception(item.get("exc_msg", "internal exception")),
                sample_idx=sample_idx,
                internal_where=item.get("where"),
                internal_exc_type=item.get("exc_type"),
                internal_exc_msg=item.get("exc_msg"),
                internal_traceback=item.get("traceback"),
                internal_context=item.get("context"),
                solution=_safe_truncate(solution, 2000),
                completion=_safe_truncate(completion, 2000),
            )
    except Exception:
        pass
    finally:
        try:
            _INTERNAL_EXCEPTION_BUFFER.set([])
        except Exception:
            pass


def _log_internal_exception(where: str, exc: BaseException, **context) -> None:
    """Backwards-compatible name used throughout this file.

    Important: This function does NOT write logs. It only records to a buffer.
    The outer `math_reward` loop is responsible for flushing to disk.
    """
    _record_internal_exception(where, exc, **context)


def _math_verify_parse(*args, **kwargs):
    """Compatibility wrapper around math_verify.parse.

    Uses parsing_timeout=None by default to avoid signal.alarm() in threaded
    environments. Can be overridden with env `MATH_VERIFY_PARSING_TIMEOUT`.
    """
    kwargs = dict(kwargs)
    kwargs.setdefault("parsing_timeout", _MATH_VERIFY_PARSING_TIMEOUT)
    expr_preview = _safe_truncate(args[0] if args else None, 800)
    try:
        return parse(*args, **kwargs)
    except TypeError as e:
        # Older math_verify versions may not support parsing_timeout kwarg.
        if "parsing_timeout" in str(e):
            kwargs.pop("parsing_timeout", None)
            try:
                return parse(*args, **kwargs)
            except TimeoutError:
                raise
            except Exception as retry_exc:
                _log_internal_exception(
                    "_math_verify_parse/retry_without_timeout",
                    retry_exc,
                    expr=expr_preview,
                )
                return []
        _log_internal_exception(
            "_math_verify_parse/type_error",
            e,
            expr=expr_preview,
        )
        return []
    except TimeoutError:
        raise
    except Exception as e:
        _log_internal_exception(
            "_math_verify_parse/parse",
            e,
            expr=expr_preview,
        )
        return []


def _safe_truncate(value, max_chars: int):
    try:
        s = value if isinstance(value, str) else repr(value)
    except Exception:
        s = "<unreprable>"
    if max_chars is not None and max_chars > 0 and len(s) > max_chars:
        return s[:max_chars] + f"...<truncated {len(s) - max_chars} chars>"
    return s


def _append_json_line(path: str, obj: dict) -> None:
    """Append one JSON object per line.

    Note: file is named *.json for convenience, but content is JSONL.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        line = json.dumps(obj, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        # Never break training due to exception logging.
        pass


def _log_exception(where: str, exc: BaseException, **context) -> None:
    if not log_exceptions:
        return
    try:
        # Keep individual records reasonably small to reduce the chance of interleaved writes.
        max_chars = int(os.environ.get("MATH_REWARD_EXCEPTION_MAX_CHARS", "2000"))
        tb = traceback.format_exc()

        record = {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "unix": time.time(),
            "pid": os.getpid(),
            "run_id": _RUN_ID,
            "where": where,
            "exc_type": type(exc).__name__,
            "exc_msg": _safe_truncate(str(exc), max_chars),
            "traceback": _safe_truncate(tb, max_chars * 3),
            "context": {k: _safe_truncate(v, max_chars) for k, v in context.items()},
            "env": {
                "MATH_REWARD_LOG_ROOT": os.environ.get("MATH_REWARD_LOG_ROOT"),
                "MATH_REWARD_RUN_ID": os.environ.get("MATH_REWARD_RUN_ID"),
            },
            "platform": {
                "python": sys.version.split()[0],
                "system": platform.system(),
                "release": platform.release(),
            },
        }

        exceptions_path = os.path.join(_RUN_LOG_DIR, "exceptions.json")
        _append_json_line(exceptions_path, record)
    except Exception:
        pass


def _make_run_id() -> str:
    """Create a filesystem-safe identifier for this Python process run.

    If you want multiple processes to share one run id (e.g. DDP), set
    env var `MATH_REWARD_RUN_ID` in the launcher.
    """
    forced = os.environ.get("MATH_REWARD_RUN_ID")
    if forced:
        return forced

    now = datetime.datetime.now(datetime.timezone.utc)
    # e.g. 2026-01-05T12-34-56Z_pid12345
    ts = now.strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"{ts}_pid{os.getpid()}"


# Root directory for math reward logs. Override if needed.
_LOG_ROOT_DIR = os.environ.get(
    "MATH_REWARD_LOG_ROOT",
    "/data/wenshuiluo/data/log/math_RL_results",
)

# Per-run directory so a single run is grouped together.
_RUN_ID = _make_run_id()
_RUN_LOG_DIR = os.path.join(_LOG_ROOT_DIR, _RUN_ID)

def _write_json_record_by_ts(output_dir: str, record: dict) -> None:
    """Write one pretty JSON file per call.

    Filename is derived from the record's `ts` in a filesystem-safe way.
    A pid suffix is added to avoid collisions across processes.
    """
    os.makedirs(output_dir, exist_ok=True)

    ts = str(record.get("ts") or "")
    # Make it filesystem-safe: remove/replace characters like ':' and '+'.
    ts_safe = (
        ts.replace(":", "-")
        .replace("+", "_")
        .replace("Z", "Z")
        .replace(" ", "_")
    )
    if not ts_safe:
        ts_safe = str(int(time.time()))

    pid = record.get("pid")
    suffix = f"_pid{pid}" if pid is not None else ""
    file_name = f"{ts_safe}{suffix}.json"
    abs_path = os.path.abspath(os.path.join(output_dir, file_name))

    tmp_path = abs_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as wf:
        json.dump(record, wf, ensure_ascii=False, indent=2)
        wf.write("\n")
    os.replace(tmp_path, abs_path)

    # Best-effort cleanup of legacy lock file name from previous logging mode.
    try:
        legacy_lock = os.path.join(output_dir, "math_reward_log.json.lock")
        if os.path.exists(legacy_lock):
            os.remove(legacy_lock)
    except Exception as e:
        _log_exception("_write_json_record_by_ts/cleanup", e, output_dir=output_dir)


# ============================================================
# Utility Functions
# ============================================================

def extract_last_boxed(text: str) -> Optional[str]:
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[-1].strip() if matches else None


def extract_final_answer(text: str) -> Optional[str]:
    """AIME-style extraction: try boxed, bold, display math, or inline 'is (...)'."""
    if not text or not isinstance(text, str):
        return None

    boxed = extract_last_boxed(text)
    if boxed:
        return boxed

    # # **answer**
    # m = re.findall(r"\*\*(.*?)\*\*", text, re.DOTALL)
    # if m:
    #     return m[-1].strip()

    # # \[ ... \]
    # m = re.findall(r"\\\[\s*(.*?)\s*\\\]", text, re.DOTALL)
    # if m:
    #     return m[-1].strip()

    # # is \( ... \)
    # m = re.findall(r"is\s+\\\((.*?)\\\)", text, re.DOTALL | re.IGNORECASE)
    # if m:
    #     return m[-1].strip()

    return None


def _is_completely_wrapped_by_text(input_string: str) -> Optional[str]:
    m = re.match(r"^\\text\{(.*)\}$", input_string)
    if not m:
        return None
    extracted = m.group(1)
    extracted = extracted.replace("(", "").replace(")", "")
    return extracted


def math_answer_cleaning(answer: str) -> str:
    """Ported/trimmed from eval_release/evaluate_aime.py."""
    if answer is None:
        return ""
    answer = str(answer)
    # Some datasets accidentally include Python-escaped control characters.
    # E.g. "\tan" becomes a literal tab (\t) + "an" when not written as a raw string.
    # Convert tabs back into a LaTeX command prefix.
    answer = answer.replace("\t", "\\t")
    # Some pipelines double-escape LaTeX, yielding strings like `\\cos` instead of `\cos`.
    # Collapse double-backslashes that directly precede a LaTeX command name.
    answer = re.sub(r"\\\\(?=[A-Za-z])", r"\\", answer)
    # Normalize common non-breaking space to regular space.
    answer = answer.replace("\u00a0", " ")
    extracted = _is_completely_wrapped_by_text(answer)
    answer = extracted if extracted else answer

    # common formatting noise
    answer = answer.replace(",\\!", "").replace("{,}", "").replace("\\$", "").replace("$", "")
    # common invisible spacing/approx symbols
    answer = answer.replace("\\!", "")
    answer = answer.replace("~", "")
    answer = answer.replace("dfrac{", "frac{").replace("tfrac{", "frac{")
    answer = answer.replace("\\left", "").replace("\\right", "")

    # Convert degree notation into radians so symbolic/numeric comparison works.
    # Examples:
    #   36^\circ  -> \frac{36\pi}{180}
    #   36^{\circ} -> \frac{36\pi}{180}
    # This is important for identities like \cos 36^\circ = (1+\sqrt5)/4.
    answer = re.sub(
        r"([+-]?\d+(?:\.\d+)?)\s*\^\s*\\circ",
        r"\\frac{\1\\pi}{180}",
        answer,
    )
    answer = re.sub(
        r"([+-]?\d+(?:\.\d+)?)\s*\^\s*\{\\circ\}",
        r"\\frac{\1\\pi}{180}",
        answer,
    )
    # Drop any remaining degree markers (best-effort; should be rare after conversion).
    answer = answer.replace("^\\circ", "").replace("^{\\circ}", "")
    answer = answer.replace("\\quad", "")

    # Drop units like \text{ cm}^2 or \mathrm{cm}^2 entirely (including the trailing exponent).
    answer = re.sub(r"\\,?\\(text|mathrm)\{.*?\}\s*\^\s*\{?\d+\}?", "", answer)
    answer = re.sub(r"\\(text|mathrm)\{.*?\}\s*\^\s*\{?\d+\}?", "", answer)
    # Drop remaining unit-like text wrappers.
    answer = re.sub(r"\\,\\(text|mathrm)\{.*?\}", "", answer)
    answer = re.sub(r"\\(text|mathrm)\{.*?\}", "", answer)

    answer = re.sub(r"\\,\\text\{.*?\}", "", answer)
    answer = re.sub(r"\\text\{.*?\}", "", answer)
    answer = re.sub(r"(\s\^\{-\d+\})", "", answer)
    answer = answer.replace(" ", "")
    answer = answer.replace("\n", "").replace("\\n", "")

    # Strip trailing stray quotes/primes (dataset noise like `$${2}$$'`).
    answer = re.sub(r"'+$", "", answer)

    # Treat single-element brace-wrapping like `{2}` or `\{2\}` as `2`.
    m = re.match(r"^\\?\{\s*([+-]?\d*\.?\d+(?:e[+-]?\d+)?)\s*\\?\}$", answer, re.IGNORECASE)
    if m:
        answer = m.group(1)

    # 3.54\times10^{10} -> 3.54e10
    answer = re.sub(r"([+-]?\d*\.?\d+)[\\]times10\^{([+-]?\d+)}", r"\1e\2", answer)
    answer = re.sub(r"([+-]?\d*\.?\d+)[\\]times10\^([+-]?\d+)", r"\1e\2", answer)
    answer = re.sub(r"(\d+)\^{(\d+)}", r"\1^\2", answer)
    answer = re.sub(r"10\^\{(-?\d+)\}", r"1e\1", answer)
    # Only remove commas used as thousands separators (e.g., 1,000 -> 1000).
    # Do NOT remove tuple/interval separators like (0.5,1).
    answer = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", answer)

    # Normalize common log/ln shorthand that many LaTeX parsers choke on.
    # Examples:
    #   - \ln^22  -> \ln^{2}{2}
    #   - \ln^{2}\frac{a}{b} -> \ln^{2}{\frac{a}{b}}
    for fn in ("ln", "log"):
        answer = re.sub(
            rf"\\{fn}\^\s*\{{?2\}}?\s*(\\frac\{{.*?\}}\{{.*?\}})",
            rf"\\{fn}^{{2}}{{\g<1>}}",
            answer,
        )
        answer = re.sub(
            rf"\\{fn}\^\s*\{{?2\}}?\s*(\\sqrt\{{.*?\}})",
            rf"\\{fn}^{{2}}{{\g<1>}}",
            answer,
        )
        answer = re.sub(
            rf"\\{fn}\^\s*\{{?2\}}?\s*([+-]?\d+(?:\.\d+)?)",
            rf"\\{fn}^{{2}}{{\g<1>}}",
            answer,
        )
        answer = re.sub(
            rf"\\{fn}\^\s*\{{?2\}}?\s*([A-Za-z])",
            rf"\\{fn}^{{2}}{{\g<1>}}",
            answer,
        )

    answer = answer.lower()

    if answer.endswith("\\"):
        answer = answer[:-1]

    # f(x)=... or x=... -> take RHS (short LHS)
    # NOTE: allow numeric/function arguments like f(1987), g(2x+1), etc.
    # We only use this to decide whether to take RHS of `lhs=rhs`.
    func_pattern = r"^[a-zA-Z_]\w*\([^=]*\)$"
    if "=" in answer:
        lhs = answer.split("=", 1)[0]
        if re.match(func_pattern, lhs) or len(lhs) <= 3:
            answer = answer.split("=", 1)[1]

    return answer


def _split_top_level_commas(s: str) -> List[str]:
    """Split a string by top-level commas, respecting (), [], {} nesting."""
    parts: List[str] = []
    buf: List[str] = []
    depth_paren = 0
    depth_brack = 0
    depth_brace = 0
    for ch in s:
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren = max(0, depth_paren - 1)
        elif ch == "[":
            depth_brack += 1
        elif ch == "]":
            depth_brack = max(0, depth_brack - 1)
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace = max(0, depth_brace - 1)

        if ch == "," and depth_paren == 0 and depth_brack == 0 and depth_brace == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return parts


def _strip_math_delimiters(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("$$") and s.endswith("$$"):
        s = s[2:-2].strip()
    elif s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    # LaTeX inline/display math delimiters.
    elif s.startswith("\\(") and s.endswith("\\)"):
        # IMPORTANT: keep the parentheses; only remove the escaping backslashes.
        # This matters for tuple/interval answers like `\(a,b\)`.
        s = "(" + s[2:-2].strip() + ")"
    elif s.startswith("\\[") and s.endswith("\\]"):
        s = s[2:-2].strip()
    return s


def _is_tuple_like(s: str) -> bool:
    s = _strip_math_delimiters(s)
    s = s.replace("\\left", "").replace("\\right", "").strip()
    return (
        (s.startswith("(") and s.endswith(")"))
        or (s.startswith("[") and s.endswith("]"))
        or (s.startswith("{") and s.endswith("}"))
    )


def _parse_percent_to_float(expr: str) -> Optional[float]:
    """Parse a percent-like expression into a float probability.

    Examples:
      - '33.3%' -> 0.333
      - '33\frac{1}{3}\%' -> 0.333333...
    """
    if not expr:
        return None
    s = _strip_math_delimiters(expr)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.strip()

    # Some datasets wrap answers in parentheses or \( \). Strip one extra pair.
    if s.startswith("(") and s.endswith(")") and ("%" in s or "\\%" in s):
        s = s[1:-1].strip()

    if s.endswith("\\%"):
        s_num = s[:-2]
    elif s.endswith("%"):
        s_num = s[:-1]
    else:
        return None

    s_num = s_num.strip()
    if not s_num:
        return None

    # Mixed-number percent like `33\frac{1}{3}\%` typically means `33 + 1/3` percent,
    # but many LaTeX parsers interpret `33\frac{1}{3}` as multiplication.
    # Normalize only for the leading integer part.
    s_num = re.sub(r"^([+-]?\d+)\\frac\{", r"\1+\\frac{", s_num)

    parsed = try_parse(s_num)
    if not parsed:
        return None
    try:
        return float(parsed[0]) / 100.0
    except Exception as e:
        _log_internal_exception(
            "_parse_percent_to_float/float",
            e,
            expr=_safe_truncate(expr, 500),
            s_num=_safe_truncate(s_num, 500),
            parsed0=_safe_truncate(parsed[0] if parsed else None, 500),
        )
        return None


def _percent_tolerance(expr: str) -> float:
    """Choose a tolerance based on decimal places in the percent string.

    If the candidate uses k digits after the decimal in percent-space, rounding
    error is at most 0.5 * 10^{-k} percent = 0.5*10^{-k-2} in probability.
    Default to k=0 -> 0.005.
    """
    s = _strip_math_delimiters(expr)
    s = s.replace("\\%", "%")
    if "%" in s:
        s = s.split("%", 1)[0]
    m = re.search(r"\.(\d+)", s)
    k = len(m.group(1)) if m else 0
    return 0.5 * (10 ** (-(k + 2)))


def _try_parse_tuple_elements(expr: str) -> Optional[List[str]]:
    s = _strip_math_delimiters(expr)
    s = s.replace("\\left", "").replace("\\right", "").strip()
    if not _is_tuple_like(s):
        return None
    inner = s[1:-1].strip()
    if not inner:
        return None
    parts = _split_top_level_commas(inner)
    if len(parts) < 2:
        return None
    return parts


def is_malformed(expr: str) -> bool:
    if not expr:
        return True
    if re.search(r"[\+\-\*/^]$", expr):  # ends with operator
        return True
    if expr.count("{") != expr.count("}") or expr.count("(") != expr.count(")"):
        return True
    return False


def try_parse(expression: str):
    """Try to parse a math expression robustly."""
    if not expression or not isinstance(expression, str):
        return []
    expr = expression.strip()
    # Normalize non-breaking spaces.
    expr = expr.replace("\u00a0", " ")
    # Normalize ratio notation like `4:3` -> `4/3` (including with spaces).
    expr = re.sub(r"(?<=\d)\s*:\s*(?=\d)", "/", expr)
    # Fix common dataset/model glitch: missing backslash in LaTeX commands.
    # E.g. `frac{x}{y}` (often coming from `$$frac{x}{y}$$`).
    expr = re.sub(r"(?<!\\)frac\{", r"\\frac{", expr)

    # If it's a plain scientific-notation literal, prefer sympify directly.
    # Some LaTeX/regex extractors may treat the `e` as a symbol/constant and produce
    # a non-numeric expression (breaking numeric tolerance checks).
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)e[+-]?\d+", expr.lower()):
        try:
            return [sympify(expr.lower())]
        except Exception as e:
            _log_internal_exception(
                "try_parse/scinotation_sympify",
                e,
                expr=_safe_truncate(expr, 500),
            )
            pass
    if is_malformed(expr):
        return []

    def _is_tuple_like_obj(obj) -> bool:
        # SymPy may parse things like (1,2) into SympyTuple, which is not safe to
        # place inside Eq(...) or symbolic simplification chains.
        return isinstance(obj, (tuple, SympyTuple))

    def _contains_tuple_like_obj(obj, _seen: Optional[set[int]] = None) -> bool:
        """True when object itself or any nested arg is tuple-like."""
        if _is_tuple_like_obj(obj):
            return True

        if _seen is None:
            _seen = set()
        oid = id(obj)
        if oid in _seen:
            return False
        _seen.add(oid)

        # Fast path for SymPy expressions.
        try:
            if hasattr(obj, "has") and obj.has(SympyTuple):
                return True
        except Exception:
            pass

        # Generic recursive descent through expression args.
        try:
            for arg in getattr(obj, "args", ()):
                if _contains_tuple_like_obj(arg, _seen):
                    return True
        except Exception:
            pass
        return False

    def _sanitize_parsed_output(parsed_obj, where: str):
        if parsed_obj is None:
            return []
        parsed_list = list(parsed_obj) if isinstance(parsed_obj, (list, tuple)) else [parsed_obj]
        if not parsed_list:
            return []
        for item in parsed_list:
            if _contains_tuple_like_obj(item):
                _log_internal_exception(
                    where,
                    Exception("Filtered tuple-like parse result"),
                    expr=_safe_truncate(expr, 800),
                    parsed=_safe_truncate(item, 500),
                )
                return []
        return parsed_list

    # Sympy parsing with implicit multiplication helps with strings like:
    #   - 60t + 90(3.5-t)
    #   - 2x(x+1)
    # without requiring explicit '*'.
    def _sympy_parse_implicit(s: str):
        try:
            from sympy.parsing.sympy_parser import (
                parse_expr,
                standard_transformations,
                implicit_multiplication_application,
                convert_xor,
            )

            transformations = standard_transformations + (
                implicit_multiplication_application,
                convert_xor,
            )
            # `evaluate=False` avoids eager evaluation that can sometimes mask structure.
            # Provide factorial2 so we can parse strings rewritten from `n!!`.
            return parse_expr(
                s,
                transformations=transformations,
                evaluate=False,
                local_dict={"factorial2": factorial2},
            )
        except TimeoutError:
            raise
        except Exception as e:
            _log_internal_exception(
                "try_parse/sympy_parse_implicit",
                e,
                s=_safe_truncate(s, 800),
            )
            return None

    # Special-case: double factorial notation `n!!` is not standard LaTeX/SymPy.
    # Rewrite to factorial2(n) and parse via SymPy to avoid returning -1 and to
    # prevent downstream verify()/simplify() from getting stuck on malformed strings.
    if "!!" in expr:
        expr_df = expr
        expr_df = expr_df.replace("\\cdot", "*").replace("\\times", "*")
        # Handle common forms like 23!! or 11!! inside larger expressions.
        expr_df = re.sub(r"(\d+)\s*!!", r"factorial2(\1)", expr_df)
        parsed_df = _sympy_parse_implicit(expr_df)
        if parsed_df is not None:
            if _is_tuple_like_obj(parsed_df):
                return []
            # If it's purely numeric, fold it to an Integer so downstream logic
            # can use the normal numeric equivalence/verify paths.
            try:
                if hasattr(parsed_df, "free_symbols") and not parsed_df.free_symbols:
                    # Avoid evaluating extremely large factorial2 arguments.
                    huge = False
                    try:
                        for node in parsed_df.atoms(factorial2):
                            if not node.args:
                                continue
                            arg0 = node.args[0]
                            if getattr(arg0, "is_integer", False) and getattr(arg0, "is_number", False):
                                try:
                                    if int(arg0) > 500:
                                        huge = True
                                        break
                                except Exception as e:
                                    _log_internal_exception(
                                        "try_parse/factorial2_int_check",
                                        e,
                                        arg0=_safe_truncate(arg0, 300),
                                    )
                                    continue
                    except Exception as e:
                        _log_internal_exception(
                            "try_parse/factorial2_atoms",
                            e,
                            expr=_safe_truncate(expr_df, 800),
                        )
                        huge = False

                    if (not huge) and getattr(parsed_df, "is_integer", False):
                        return [Integer(int(parsed_df))]
            except Exception as e:
                _log_internal_exception(
                    "try_parse/factorial2_numeric_fold",
                    e,
                    expr=_safe_truncate(expr_df, 800),
                )
                pass
            return [parsed_df]

    def _normalize_trig_shorthand_latex(s: str) -> str:
        """Normalize common LaTeX function shorthands to improve parsers.

        Handles cases like:
          - \\sec^2 x   -> \\sec^{2}{x}
          - \\sin2x    -> \\sin{2x}
          - \\cos 2x   -> \\cos{2x}
          - \\ln^22    -> \\ln^{2}{2}   (i.e., (ln 2)^2)
          - \\ln^{2}\\frac{a}{b} -> \\ln^{2}{\\frac{a}{b}}
        """
        if not s:
            return s
        out = s
        # Remove display delimiters if they leaked in.
        out = out.replace("\\[", "").replace("\\]", "")

        # Normalize log base shorthand when spaces are stripped.
        # Examples:
        #   - \log_25   -> \log_{2}{5}
        #   - \log_2x   -> \log_{2}{x}
        #   - \log_2(1+x) -> \log_{2}{(1+x)}
        #   - \log_2\frac{a}{b} -> \log_{2}{\frac{a}{b}}
        # Do not touch already-braced forms like \log_{2}5.
        out = re.sub(
            r"\\log_\{",  # sentinel: keep existing braced base
            r"\\log_{",
            out,
        )
        # Base followed by a LaTeX command argument.
        out = re.sub(
            r"\\log_([0-9]+)(\\\\[A-Za-z]+\{.*?\})",
            lambda m: f"\\log_{{{m.group(1)}}}{{{m.group(2)}}}",
            out,
            flags=re.DOTALL,
        )
        # Base followed by a parenthesized argument.
        out = re.sub(
            r"\\log_([0-9]+)(\([^)]*\))",
            lambda m: f"\\log_{{{m.group(1)}}}{{{m.group(2)}}}",
            out,
        )
        # Base followed by a simple token (digit(s) or single letter).
        out = re.sub(
            r"\\log_([0-9]+)([A-Za-z]|[0-9]+)",
            lambda m: f"\\log_{{{m.group(1)}}}{{{m.group(2)}}}",
            out,
        )
        # Finally convert \log_{b}{x} into latex2sympy-friendly \log_{b}x form if needed.
        out = re.sub(r"\\log_\{([0-9]+)\}\{(.*?)\}", r"\\log_{\1}{\2}", out)

        # Normalize powered functions without braces, e.g.
        #   \sec^2 x, \sec^2x, \sec^{2}x
        #   \ln^22,  \ln^2 2
        #   \ln^{2}\frac{a}{b}
        for fn in ("sin", "cos", "tan", "sec", "csc", "cot", "ln", "log"):
            # Single-letter variable argument.
            out = re.sub(
                rf"\\{fn}\^\s*\{{?2\}}?\s*([A-Za-z])",
                lambda m, fn=fn: f"\\{fn}^{{2}}{{{m.group(1)}}}",
                out,
            )
            # Numeric argument (possibly multi-digit / decimal).
            out = re.sub(
                rf"\\{fn}\^\s*\{{?2\}}?\s*([+-]?\d+(?:\.\d+)?)",
                lambda m, fn=fn: f"\\{fn}^{{2}}{{{m.group(1)}}}",
                out,
            )
            # Simple \frac{...}{...} argument (common in datasets).
            out = re.sub(
                rf"\\{fn}\^\s*\{{?2\}}?\s*(\\frac\{{.*?\}}\{{.*?\}})",
                lambda m, fn=fn: f"\\{fn}^{{2}}{{{m.group(1)}}}",
                out,
                flags=re.DOTALL,
            )
            # Simple \sqrt{...} argument.
            out = re.sub(
                rf"\\{fn}\^\s*\{{?2\}}?\s*(\\sqrt\{{.*?\}})",
                lambda m, fn=fn: f"\\{fn}^{{2}}{{{m.group(1)}}}",
                out,
                flags=re.DOTALL,
            )

        # Wrap direct arguments like \sin2x, \cos2x, \tan2x, \ln2 (no braces/paren)
        # Also handle direct LaTeX-structure arguments like \cos\frac{...}{...}.
        out = re.sub(
            r"\\(sin|cos|tan|sec|csc|cot|ln|log)(?!\s*\{|\s*\()(\\frac\{.*?\}\{.*?\})",
            r"\\\1{\2}",
            out,
            flags=re.DOTALL,
        )
        out = re.sub(
            r"\\(sin|cos|tan|sec|csc|cot|ln|log)(?!\s*\{|\s*\()(\\sqrt\{.*?\})",
            r"\\\1{\2}",
            out,
            flags=re.DOTALL,
        )
        out = re.sub(
            r"\\(sin|cos|tan|sec|csc|cot|ln|log)(?!\s*\{|\s*\()(\d+[A-Za-z]|\d+)",
            r"\\\1{\2}",
            out,
        )
        # Also handle the common case where spaces were stripped: \tanx -> \tan{x}.
        out = re.sub(
            r"\\(sin|cos|tan|sec|csc|cot|ln|log)(?!\s*\{|\s*\()([A-Za-z])",
            r"\\\1{\2}",
            out,
        )
        out = re.sub(
            r"\\(sin|cos|tan|sec|csc|cot|ln|log)\s+(\d+[A-Za-z]|\d+)",
            r"\\\1{\2}",
            out,
        )
        out = re.sub(
            r"\\(sin|cos|tan|sec|csc|cot|ln|log)\s+([A-Za-z])",
            r"\\\1{\2}",
            out,
        )

        return out

    # Prefer parsing the *entire* LaTeX expression when possible.
    # `math_verify.parse(..., extraction_mode="first_match")` may only extract the first
    # parseable sub-expression (e.g. `\frac{1}{2}` from `-\frac{1}{2}\cdot\ln^{2}...`).
    def _looks_like_latex(s: str) -> bool:
        if "\\" in s:
            return True
        if any(tok in s for tok in ("\\frac", "\\sqrt", "^", "_")):
            return True
        return False

    if _looks_like_latex(expr):
        latex_expr = expr
        # Strip common math delimiters.
        if latex_expr.startswith("$$") and latex_expr.endswith("$$"):
            latex_expr = latex_expr[2:-2].strip()
        elif latex_expr.startswith("$") and latex_expr.endswith("$"):
            latex_expr = latex_expr[1:-1].strip()

        latex_expr = _normalize_trig_shorthand_latex(latex_expr)

        try:
            # Handle LaTeX equations explicitly: latex2sympy may not accept '=' directly.
            if "=" in latex_expr and latex_expr.count("=") == 1:
                lhs_s, rhs_s = (p.strip() for p in latex_expr.split("=", 1))
                if lhs_s and rhs_s:
                    lhs_n = normalize_latex(lhs_s, NormalizationConfig())
                    rhs_n = normalize_latex(rhs_s, NormalizationConfig())
                    lhs_e = latex2sympy(lhs_n)
                    rhs_e = latex2sympy(rhs_n)
                    if _contains_tuple_like_obj(lhs_e) or _contains_tuple_like_obj(rhs_e):
                        return []
                    return [Eq(lhs_e, rhs_e)]

            # Normalize to help latex2sympy handle more variants.
            normalized = normalize_latex(latex_expr, NormalizationConfig())
            return _sanitize_parsed_output(latex2sympy(normalized), "try_parse/latex2sympy_tuple")
        except TimeoutError:
            raise
        except Exception as e:
            _log_internal_exception(
                "try_parse/latex2sympy",
                e,
                latex_expr=_safe_truncate(latex_expr, 800),
            )
            # Fall back to extraction-based parsing below.
            pass

    if "=" in expr and not expr.strip().startswith("\\") and expr.count("=") == 1:
        lhs_s, rhs_s = (p.strip() for p in expr.split("=", 1))
        if lhs_s and rhs_s:
            # Try implicit-multiplication parse first (handles `60t`, `90(…)`).
            lhs_i = _sympy_parse_implicit(lhs_s)
            rhs_i = _sympy_parse_implicit(rhs_s)
            if lhs_i is not None and rhs_i is not None:
                if _contains_tuple_like_obj(lhs_i) or _contains_tuple_like_obj(rhs_i):
                    return []
                try:
                    return [Eq(lhs_i, rhs_i)]
                except TimeoutError:
                    raise
                except Exception as e:
                    _log_internal_exception(
                        "try_parse/equation_implicit_eq",
                        e,
                        lhs=_safe_truncate(lhs_s, 400),
                        rhs=_safe_truncate(rhs_s, 400),
                    )
                    return []
            try:
                lhs, rhs = sympify(lhs_s), sympify(rhs_s)
                if _contains_tuple_like_obj(lhs) or _contains_tuple_like_obj(rhs):
                    return []
                return [Eq(lhs, rhs)]
            except TimeoutError:
                raise
            except Exception as e:
                _log_internal_exception(
                    "try_parse/equation_sympify",
                    e,
                    lhs=_safe_truncate(lhs_s, 400),
                    rhs=_safe_truncate(rhs_s, 400),
                )
                pass

    # For plain (non-LaTeX) expressions, prefer SymPy implicit-multiplication parsing
    # when we see patterns like `3(64.5+18)` or `)(`.
    # This avoids extraction-based parsing returning an unparsed string.
    if "\\" not in expr and (
        re.search(r"(?<=\d)\(", expr) is not None or re.search(r"\)\(", expr) is not None
    ):
        implicit = _sympy_parse_implicit(expr)
        if implicit is not None:
            return _sanitize_parsed_output(implicit, "try_parse/implicit_tuple")

    parsed = _math_verify_parse(expr, extraction_mode="first_match")
    if parsed:
        return _sanitize_parsed_output(parsed, "try_parse/math_verify_first_match_tuple")

    # Retry with normalization
    parsed = _math_verify_parse(
        expr,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    boxed="all",
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=True,
            )
        ],
        extraction_mode="first_match",
    )
    if parsed:
        return _sanitize_parsed_output(parsed, "try_parse/math_verify_normalized_tuple")

    try:
        implicit = _sympy_parse_implicit(expr)
        if implicit is not None:
            return _sanitize_parsed_output(implicit, "try_parse/final_implicit_tuple")
        return _sanitize_parsed_output(sympify(expr), "try_parse/final_sympify_tuple")
    except Exception as e:
        _log_internal_exception(
            "try_parse/final_sympify",
            e,
            expr=_safe_truncate(expr, 800),
        )
        return []


def is_numeric_equivalent(a, b, tol=1e-6):
    try:
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    except Exception as e:
        _log_internal_exception(
            "is_numeric_equivalent",
            e,
            a=_safe_truncate(a, 300),
            b=_safe_truncate(b, 300),
            tol=tol,
        )
        return False


def is_symbolically_equivalent(expr1, expr2):
    from sympy import expand, simplify, trigsimp, logcombine, expand_log, expand_power_exp, cancel, factorial, factorial2, Pow, Sum

    def _contains_tuple_like_symbolic(expr, _seen: Optional[set[int]] = None) -> bool:
        if isinstance(expr, (tuple, SympyTuple)):
            return True
        if _seen is None:
            _seen = set()
        oid = id(expr)
        if oid in _seen:
            return False
        _seen.add(oid)
        try:
            if hasattr(expr, "has") and expr.has(SympyTuple):
                return True
        except Exception:
            pass
        try:
            for arg in getattr(expr, "args", ()):
                if _contains_tuple_like_symbolic(arg, _seen):
                    return True
        except Exception:
            pass
        return False

    if _contains_tuple_like_symbolic(expr1) or _contains_tuple_like_symbolic(expr2):
        _log_internal_exception(
            "is_symbolically_equivalent/tuple_like_short_circuit",
            Exception("Tuple-like expression not supported in symbolic equivalence"),
            expr1=_safe_truncate(expr1, 500),
            expr2=_safe_truncate(expr2, 500),
        )
        return False
    
    # Early check: if expressions are too large, skip expensive symbolic operations
    def _is_too_large_for_symbolic(expr) -> bool:
        """Check if expression is too large/complex for symbolic simplification."""
        try:
            if not hasattr(expr, 'atoms'):
                return False

            # 含有双阶乘直接开启numeric判断即可
            try:
                if expr.atoms(factorial2):
                    return True
            except Exception:
                pass
            
            # Check for Sum expressions (like \sum_{k=1}^{n})
            # These are extremely slow to expand/simplify
            try:
                if expr.atoms(Sum):
                    _log_internal_exception(
                        "is_symbolically_equivalent/_is_too_large_for_symbolic",
                        Exception("Detected Sum expression, skipping symbolic simplification"),
                        expr=_safe_truncate(expr, 300),
                    )
                    return True
            except Exception:
                pass
            
            # Check for large powers
            for atom in expr.atoms(Pow):
                if len(atom.args) >= 2:
                    base, exp = atom.args[0], atom.args[1]
                    # Check if it's a large integer power
                    if hasattr(base, 'is_number') and hasattr(exp, 'is_number'):
                        if base.is_number and exp.is_number:
                            try:
                                b_val = abs(float(base))
                                e_val = abs(float(exp))
                                # Avoid expanding things like 6^10, 100^5, etc.
                                if b_val > 5 and e_val > 5:
                                    return True
                                if b_val > 10 and e_val > 3:
                                    return True
                                if e_val > 1000:
                                    return True
                            except:
                                pass
            
            # Check for factorials with large arguments
            for atom in expr.atoms(factorial):
                if atom.args:
                    arg0 = atom.args[0]
                    if hasattr(arg0, 'is_integer') and hasattr(arg0, 'is_number'):
                        if arg0.is_integer and arg0.is_number:
                            try:
                                if int(arg0) > 20:
                                    return True
                            except:
                                pass
            return False
        except Exception as e:
            _log_internal_exception(
                "is_symbolically_equivalent/_is_too_large_for_symbolic",
                e,
                expr=_safe_truncate(expr, 300),
            )
            return True  # If we can't check, assume it's too large to be safe
    
    try:
        # Skip expensive symbolic operations for large expressions
        if _is_too_large_for_symbolic(expr1) or _is_too_large_for_symbolic(expr2):
            _log_internal_exception(
                "is_symbolically_equivalent/skipped_large_expr",
                Exception("Skipped symbolic equivalence check for large expression"),
                expr1=_safe_truncate(expr1, 300),
                expr2=_safe_truncate(expr2, 300),
            )
            # Fall through to numeric test below
            raise Exception("Expression too large for symbolic simplification")
        
        # Equation equivalence: accept same equation up to swapping sides.
        if isinstance(expr1, Eq) and isinstance(expr2, Eq):
            d1 = simplify(expr1.lhs - expr1.rhs)
            d2 = simplify(expr2.lhs - expr2.rhs)
            diff = simplify(cancel(expand(d1 - d2)))
            if diff == 0:
                return True
            diff_swapped = simplify(cancel(expand(d1 + d2)))
            if diff_swapped == 0:
                return True

        s1 = trigsimp(simplify(logcombine(expand_log(expand_power_exp(expand(expr1, deep=True))))))
        s2 = trigsimp(simplify(logcombine(expand_log(expand_power_exp(expand(expr2, deep=True))))))
        diff = simplify(cancel(expand(s1 - s2)))
        if diff == 0:
            return True
    except Exception as e:
        _log_internal_exception(
            "is_symbolically_equivalent/symbolic",
            e,
            expr1=_safe_truncate(expr1, 500),
            expr2=_safe_truncate(expr2, 500),
        )
        pass

    # Fallback numeric test
    try:
        import random
        vars_ = list(expr1.free_symbols.union(expr2.free_symbols))
        if vars_:
            # Avoid huge evaluation cost for expressions containing factorial/factorial2.
            # Use tiny substitutions and reject large factorial arguments after substitution.
            max_sub = 10
            try:
                if (
                    (hasattr(expr1, "atoms") and (expr1.atoms(factorial) or expr1.atoms(factorial2)))
                    or (hasattr(expr2, "atoms") and (expr2.atoms(factorial) or expr2.atoms(factorial2)))
                ):
                    max_sub = 3
            except Exception:
                max_sub = 3

            def _has_huge_factorial_arg(e) -> bool:
                try:
                    if not hasattr(e, "atoms"):
                        return False
                    for f in e.atoms(factorial):
                        if not getattr(f, "args", None):
                            continue
                        a0 = f.args[0]
                        if getattr(a0, "is_integer", False) and getattr(a0, "is_number", False):
                            try:
                                if int(a0) > 50:
                                    return True
                            except Exception:
                                continue
                    for f in e.atoms(factorial2):
                        if not getattr(f, "args", None):
                            continue
                        a0 = f.args[0]
                        if getattr(a0, "is_integer", False) and getattr(a0, "is_number", False):
                            try:
                                if int(a0) > 200:
                                    return True
                            except Exception:
                                continue
                    return False
                except Exception:
                    return True

            for _ in range(8):
                # Not use float first
                subs = {v: random.randint(1, max_sub) for v in vars_}
                e1 = expr1.subs(subs)
                e2 = expr2.subs(subs)
                if _has_huge_factorial_arg(e1) or _has_huge_factorial_arg(e2):
                    return False
                v1 = float(e1.evalf())
                v2 = float(e2.evalf())
                if not math.isclose(v1, v2, rel_tol=1e-6, abs_tol=1e-6):
                    return False
            return True
    except Exception as e:
        _log_internal_exception(
            "is_symbolically_equivalent/numeric_fallback",
            e,
            expr1=_safe_truncate(expr1, 500),
            expr2=_safe_truncate(expr2, 500),
        )
        pass
    return False


# ============================================================
# Accuracy Reward (factored out from original)
# ============================================================

def _accuracy_reward_from_clean(candidate_clean: str, solution_clean: str) -> Optional[float]:
    """Core accuracy logic operating on already-cleaned strings."""
    if candidate_clean and solution_clean and candidate_clean == solution_clean:
        return 1.0
    if abs(len(candidate_clean) - len(solution_clean)) > 150:
        return -1.0
    
    def _decimal_abs_tolerance(s: str) -> Optional[float]:
        """Infer an absolute tolerance from a plain decimal representation.

        For `0.000055` (6 decimals) we return 1e-6, treating the last shown
        decimal place as the precision unit.
        """
        if not s:
            return None
        x = _strip_math_delimiters(s)
        x = x.replace("\\left", "").replace("\\right", "")
        x = x.strip().lower()
        # Exclude scientific notation.
        if "e" in x:
            return None
        # Keep only a plain decimal like -12.34 or 0.000055
        m = re.fullmatch(r"[+-]?(?:\d+)?\.(\d+)", x)
        if not m:
            return None
        k = len(m.group(1))
        if k <= 0:
            return None
        return 10 ** (-k)

    # 0) Special-case: all real numbers.
    def _is_all_reals_symbol(s: str) -> bool:
        return (s or "").replace(" ", "") in {"\\mathbb{r}", "\\mathbb{R}", "\\mathbb{reals}", "\\mathbb{reals}"}

    def _is_all_reals_interval(s: str) -> bool:
        if not s:
            return False
        x = _strip_math_delimiters(s).replace("\\left", "").replace("\\right", "")
        x = x.replace(" ", "").lower()
        # normalize (+\infty) -> (\infty)
        x = x.replace("+\\infty", "\\infty")
        return x in {"(-\\infty,\\infty)", "(-\\infty,+\\infty)", "(-infty,infty)", "(-infty,+infty)"}

    if (_is_all_reals_symbol(solution_clean) and _is_all_reals_interval(candidate_clean)) or (
        _is_all_reals_symbol(candidate_clean) and _is_all_reals_interval(solution_clean)
    ):
        return 1.0

    def _is_numeric_arithmetic_equation(s: str) -> bool:
        """True if s looks like a purely numeric arithmetic equation a=b.

        We only enable equation-RHS fallback for this narrow case to avoid
        accepting answers like `r^2` for `x^2+y^2=r^2`.
        """
        if not s or "=" not in s:
            return False
        left, right = s.split("=", 1)
        if not left or not right:
            return False
        allowed = set("0123456789+-*/().^")
        return set(left).issubset(allowed) and set(right).issubset(allowed)

    # 0) Equation-style fallback (numeric arithmetic only).
    # Handle cases like:
    #   - candidate: `25+2=27`, gold: `27`
    #   - candidate: `27`, gold: `25+2=27`
    if "=" in candidate_clean and "=" not in solution_clean and _is_numeric_arithmetic_equation(candidate_clean):
        rhs = candidate_clean.rsplit("=", 1)[-1].strip()
        if rhs:
            rhs_parsed = try_parse(rhs)
            gold_parsed = try_parse(solution_clean)
            if rhs_parsed and gold_parsed:
                g0, a0 = gold_parsed[0], rhs_parsed[0]
                if is_numeric_equivalent(g0, a0) or is_symbolically_equivalent(g0, a0):
                    return 1.0

    if "=" in solution_clean and "=" not in candidate_clean and _is_numeric_arithmetic_equation(solution_clean):
        rhs = solution_clean.rsplit("=", 1)[-1].strip()
        if rhs:
            rhs_parsed = try_parse(rhs)
            cand_parsed = try_parse(candidate_clean)
            if rhs_parsed and cand_parsed:
                g0, a0 = rhs_parsed[0], cand_parsed[0]
                if is_numeric_equivalent(g0, a0) or is_symbolically_equivalent(g0, a0):
                    return 1.0

    # 0b) Identity-equation fallback: if one side is an equation whose LHS and RHS are
    # equivalent (identity), accept candidate matching either side.
    def _try_identity_equation_match(expr_equation: str, expr_value: str) -> bool:
        if not expr_equation or "=" not in expr_equation:
            return False
        if not expr_value:
            return False
        # Only handle single '='.
        if expr_equation.count("=") != 1:
            return False
        lhs_s, rhs_s = (p.strip() for p in expr_equation.split("=", 1))
        if not lhs_s or not rhs_s:
            return False

        lhs_p = try_parse(lhs_s)
        rhs_p = try_parse(rhs_s)
        val_p = try_parse(expr_value)
        if not lhs_p or not rhs_p or not val_p:
            return False
        lhs, rhs, val = lhs_p[0], rhs_p[0], val_p[0]

        # Ensure the equation is actually an identity (avoid accepting partials for relations).
        identity_ok = (
            is_numeric_equivalent(lhs, rhs)
            or is_symbolically_equivalent(lhs, rhs)
        )
        if not identity_ok:
            try:
                identity_ok = float(verify(lhs_p, rhs_p)) == 1.0
            except TimeoutError:
                raise
            except Exception as e:
                _log_internal_exception(
                    "_accuracy_reward_from_clean/identity_verify",
                    e,
                    equation=_safe_truncate(expr_equation, 800),
                    value=_safe_truncate(expr_value, 800),
                )
                identity_ok = False
        if not identity_ok:
            return False

        return (
            is_numeric_equivalent(val, lhs)
            or is_symbolically_equivalent(val, lhs)
            or is_numeric_equivalent(val, rhs)
            or is_symbolically_equivalent(val, rhs)
        )

    if "=" in solution_clean and "=" not in candidate_clean:
        if _try_identity_equation_match(solution_clean, candidate_clean):
            return 1.0
    if "=" in candidate_clean and "=" not in solution_clean:
        if _try_identity_equation_match(candidate_clean, solution_clean):
            return 1.0

    # 1) Percent handling (accept rounded percent answers)
    cand_pct = _parse_percent_to_float(candidate_clean)
    gold_pct = _parse_percent_to_float(solution_clean)
    if cand_pct is not None and gold_pct is not None:
        tol = max(_percent_tolerance(candidate_clean), _percent_tolerance(solution_clean), 1e-6)
        if math.isclose(cand_pct, gold_pct, rel_tol=0.0, abs_tol=tol):
            return 1.0

    # 2) Tuple/interval handling: compare element-wise
    cand_parts = _try_parse_tuple_elements(candidate_clean)
    gold_parts = _try_parse_tuple_elements(solution_clean)
    if cand_parts is not None and gold_parts is not None and len(cand_parts) == len(gold_parts):
        for cp, gp in zip(cand_parts, gold_parts):
            cp = cp.strip()
            gp = gp.strip()
            # Percent elements inside tuples
            cp_pct = _parse_percent_to_float(cp)
            gp_pct = _parse_percent_to_float(gp)
            if cp_pct is not None and gp_pct is not None:
                tol = max(_percent_tolerance(cp), _percent_tolerance(gp), 1e-6)
                if not math.isclose(cp_pct, gp_pct, rel_tol=0.0, abs_tol=tol):
                    return 0.0
                continue

            gp_parsed = try_parse(gp)
            cp_parsed = try_parse(cp)
            if not gp_parsed or not cp_parsed:
                return 0.0
            g0, a0 = gp_parsed[0], cp_parsed[0]
            if is_numeric_equivalent(g0, a0) or is_symbolically_equivalent(g0, a0):
                continue
            try:
                if float(verify(gp_parsed, cp_parsed)) == 1.0:
                    continue
            except TimeoutError:
                raise
            except Exception as e:
                _log_internal_exception(
                    "_accuracy_reward_from_clean/tuple_verify",
                    e,
                    gold=_safe_truncate(gp, 300),
                    cand=_safe_truncate(cp, 300),
                )
                pass
            return 0.0
        return 1.0

    # 2b) Vector/matrix handling: accept pmatrix/bmatrix/matrix equal to tuple.
    def _try_parse_matrix_elements(expr: str) -> Optional[List[str]]:
        s = _strip_math_delimiters(expr)
        s = s.replace("\\left", "").replace("\\right", "").strip()
        m = re.search(
            r"\\begin\{(pmatrix|bmatrix|matrix)\}(.*?)\\end\{\1\}",
            s,
            re.DOTALL,
        )
        if not m:
            return None
        inner = m.group(2).strip()
        if not inner:
            return None
        # Split rows by \\ and columns by &; flatten row-major.
        rows = re.split(r"\\\\", inner)
        parts: List[str] = []
        for row in rows:
            row = row.strip()
            if not row:
                continue
            cols = [c.strip() for c in row.split("&") if c.strip()]
            parts.extend(cols)
        return parts if parts else None

    cand_mat = _try_parse_matrix_elements(candidate_clean)
    gold_mat = _try_parse_matrix_elements(solution_clean)
    if cand_parts is not None and gold_mat is not None and len(cand_parts) == len(gold_mat):
        for cp, gp in zip(cand_parts, gold_mat):
            gp_parsed = try_parse(gp)
            cp_parsed = try_parse(cp)
            if not gp_parsed or not cp_parsed:
                return 0.0
            g0, a0 = gp_parsed[0], cp_parsed[0]
            if is_numeric_equivalent(g0, a0) or is_symbolically_equivalent(g0, a0):
                continue
            try:
                if float(verify(gp_parsed, cp_parsed)) == 1.0:
                    continue
            except TimeoutError:
                raise
            except Exception as e:
                _log_internal_exception(
                    "_accuracy_reward_from_clean/tuple_vs_matrix_verify",
                    e,
                    gold=_safe_truncate(gp, 300),
                    cand=_safe_truncate(cp, 300),
                )
                pass
            return 0.0
        return 1.0

    if gold_parts is not None and cand_mat is not None and len(gold_parts) == len(cand_mat):
        for gp, cp in zip(gold_parts, cand_mat):
            gp_parsed = try_parse(gp)
            cp_parsed = try_parse(cp)
            if not gp_parsed or not cp_parsed:
                return 0.0
            g0, a0 = gp_parsed[0], cp_parsed[0]
            if is_numeric_equivalent(g0, a0) or is_symbolically_equivalent(g0, a0):
                continue
            try:
                if float(verify(gp_parsed, cp_parsed)) == 1.0:
                    continue
            except TimeoutError:
                raise
            except Exception as e:
                _log_internal_exception(
                    "_accuracy_reward_from_clean/matrix_vs_tuple_verify",
                    e,
                    gold=_safe_truncate(gp, 300),
                    cand=_safe_truncate(cp, 300),
                )
                pass
            return 0.0
        return 1.0

    gold_parsed = try_parse(solution_clean)
    if not gold_parsed:
        return -1

    ans_parsed = try_parse(candidate_clean)
    if not ans_parsed:
        return -1

    g, a = gold_parsed[0], ans_parsed[0]

    # Fast path for double-factorial answers rewritten as factorial2(...).
    # These often cause parsers/verifiers to struggle; if both sides are numeric,
    # evaluate exactly and compare as integers to avoid slow simplify/verify.
    try:
        if (hasattr(g, "has") and g.has(factorial2)) or (hasattr(a, "has") and a.has(factorial2)):
            # If candidate is symbolic but gold is numeric, it's definitely wrong.
            if hasattr(g, "free_symbols") and hasattr(a, "free_symbols"):
                if (not g.free_symbols) and a.free_symbols:
                    return 0.0

            # Avoid evaluating extremely large factorial2 arguments.
            def _has_huge_factorial2_arg(expr_obj) -> bool:
                try:
                    for node in expr_obj.atoms(factorial2):
                        if not node.args:
                            continue
                        arg0 = node.args[0]
                        if getattr(arg0, "is_integer", False) and getattr(arg0, "is_number", False):
                            try:
                                if int(arg0) > 500:
                                    return True
                            except Exception as e:
                                _log_internal_exception(
                                    "_accuracy_reward_from_clean/factorial2_int_check",
                                    e,
                                    arg0=_safe_truncate(arg0, 300),
                                )
                                continue
                except Exception as e:
                    _log_internal_exception(
                        "_accuracy_reward_from_clean/factorial2_atoms",
                        e,
                        expr=_safe_truncate(expr_obj, 500),
                    )
                    return False
                return False

            if _has_huge_factorial2_arg(g) or _has_huge_factorial2_arg(a):
                return 0.0

            if hasattr(g, "free_symbols") and hasattr(a, "free_symbols") and (not g.free_symbols) and (not a.free_symbols):
                try:
                    return 1.0 if int(g) == int(a) else 0.0
                except Exception as e:
                    _log_internal_exception(
                        "_accuracy_reward_from_clean/factorial2_int_compare",
                        e,
                        gold=_safe_truncate(g, 500),
                        cand=_safe_truncate(a, 500),
                    )
                    return 0.0
    except TimeoutError:
        raise
    except Exception as e:
        _log_internal_exception(
            "_accuracy_reward_from_clean/factorial2_fastpath",
            e,
            gold=_safe_truncate(g, 500) if 'g' in locals() else None,
            cand=_safe_truncate(a, 500) if 'a' in locals() else None,
        )
        pass

    # Try numeric equivalence first (fast path)
    if is_numeric_equivalent(g, a):
        return 1.0

    # Numeric rounding tolerance: allow agreement within the implied decimal precision.
    # Useful for cases like `5.43e-5` vs `0.000055`.
    try:
        g_f = float(g)
        a_f = float(a)
        tol = max(_decimal_abs_tolerance(candidate_clean) or 0.0, _decimal_abs_tolerance(solution_clean) or 0.0)
        # Keep a sane floor so tiny values don't get overly strict.
        tol = max(tol, 1e-4)
        if math.isclose(g_f, a_f, rel_tol=0.0, abs_tol=tol):
            return 1.0
    except Exception as e:
        _log_internal_exception(
            "_accuracy_reward_from_clean/rounding_tolerance",
            e,
            gold=_safe_truncate(g, 500),
            cand=_safe_truncate(a, 500),
        )
        pass

    # Try to evaluate both as numbers before symbolic comparison (for things like 6^10)
    # This avoids expensive symbolic simplification when both are pure numeric expressions
    try:
        if hasattr(g, 'free_symbols') and hasattr(a, 'free_symbols'):
            if not g.free_symbols and not a.free_symbols:
                # Both are numeric - try to evaluate and compare as numbers
                try:
                    from sympy import N
                    # Use evalf with limited precision to avoid hanging on huge numbers
                    g_eval = N(g, 15)  # 15 digits of precision
                    a_eval = N(a, 15)
                    if abs(float(g_eval) - float(a_eval)) < 1e-9:
                        return 1.0
                except Exception as eval_exc:
                    _log_internal_exception(
                        "_accuracy_reward_from_clean/numeric_eval",
                        eval_exc,
                        gold=_safe_truncate(g, 500),
                        cand=_safe_truncate(a, 500),
                    )
                    pass
    except Exception as e:
        _log_internal_exception(
            "_accuracy_reward_from_clean/free_symbols_check",
            e,
            gold=_safe_truncate(g, 500),
            cand=_safe_truncate(a, 500),
        )
        pass

    # Only try symbolic equivalence if numeric approaches failed
    if is_symbolically_equivalent(g, a):
        return 1.0

    # Last resort: use verify(), but be cautious with large expressions
    def _should_skip_verify(expr) -> bool:
        """Check if expression is too risky for verify()."""
        try:
            from sympy import Pow, factorial
            if hasattr(expr, 'atoms'):
                # Check for large powers
                for atom in expr.atoms(Pow):
                    if len(atom.args) >= 2:
                        base, exp = atom.args[0], atom.args[1]
                        if hasattr(base, 'is_number') and hasattr(exp, 'is_number'):
                            if base.is_number and exp.is_number:
                                try:
                                    b_val = abs(float(base))
                                    e_val = abs(float(exp))
                                    # Skip verify for large powers
                                    if b_val > 5 and e_val > 5:
                                        return True
                                except:
                                    pass
                # Check for factorials
                for atom in expr.atoms(factorial):
                    if atom.args:
                        arg0 = atom.args[0]
                        try:
                            if int(arg0) > 20:
                                return True
                        except:
                            pass
            return False
        except:
            return True  # If we can't check, skip verify to be safe
    
    try:
        # Skip verify for expressions that are likely to hang
        if _should_skip_verify(g) or _should_skip_verify(a):
            _log_internal_exception(
                "_accuracy_reward_from_clean/verify_skipped",
                Exception("Skipped verify() for large/risky expression"),
                gold=_safe_truncate(solution_clean, 800),
                cand=_safe_truncate(candidate_clean, 800),
            )
            return None
        
        verified = verify(gold_parsed, ans_parsed)
        return float(verified)
    except TimeoutError:
        raise
    except Exception as e:
        _log_internal_exception(
            "_accuracy_reward_from_clean/verify",
            e,
            gold=_safe_truncate(solution_clean, 800),
            cand=_safe_truncate(candidate_clean, 800),
        )
        return None

def accuracy_reward(completion: str, solution: str) -> Optional[float]:
    candidate = extract_final_answer(completion) or completion
    if not re.search(r"[0-9a-zA-Z\\+\-\*/^=(){}]", candidate):
        return -1.0

    # AIME-style cleaning helps remove \text{}, units, commas, etc.
    # print(candidate, solution)
    candidate_clean = math_answer_cleaning(candidate[-300:])
    solution_clean = math_answer_cleaning(solution)

    return _accuracy_reward_from_clean(candidate_clean, solution_clean)
