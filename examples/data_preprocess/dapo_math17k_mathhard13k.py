# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Preprocess local JSONL math data to VERL RLHF parquet format.

Input JSONL example fields (common):
- problem
- answer
- prompt  (already chat-templated string; not used directly)
- id
- dataset_from

Output schema (VERL RLHF):
- data_source
- prompt: [{"role": "user", "content": "..."}]
- ability
- reward_model: {"style": "rule", "ground_truth": "..."}
- extra_info
- uid
"""

import argparse
import json
import os
import random
import re
from copy import deepcopy

import datasets

DEFAULT_INPUT = "/Users/shengke/Downloads/DAPO_Math17k_and_Math_hard_13k.jsonl"
DEFAULT_OUTPUT_DIR = "./dapo_math17k_mathhard13k_parquet"
DEFAULT_INSTRUCTION = r"Please place your final answer inside \boxed{}."


def _extract_user_from_templated_prompt(prompt: str) -> str:
    """Best-effort extraction for prompts like:
    <|im_start|>user
    ...
    <|im_end|>
    """
    if not isinstance(prompt, str) or not prompt:
        return ""
    match = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", prompt, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return prompt.strip()


def _build_user_content(problem: str, raw_prompt: str, instruction: str) -> str:
    content = (problem or "").strip()
    if not content:
        content = _extract_user_from_templated_prompt(raw_prompt)

    instruction = (instruction or "").strip()
    if instruction and instruction not in content:
        content = f"{content}\n\n{instruction}".strip()
    return content


def _build_ground_truth(answer, solution_cleaned, solution):
    for value in (answer, solution_cleaned, solution):
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    raise ValueError("Missing ground truth: expected one of answer / solution_cleaned / solution.")


def _normalize_example(raw: dict, idx: int, instruction: str) -> dict:
    problem = raw.get("problem")
    raw_prompt = raw.get("prompt")
    answer = raw.get("answer")
    solution_cleaned = raw.get("solution_cleaned")
    solution = raw.get("solution")

    user_content = _build_user_content(problem=problem, raw_prompt=raw_prompt, instruction=instruction)
    if not user_content:
        raise ValueError(f"Empty prompt content at index={idx}")


    data_source = str(raw.get("dataset_from") or "custom_math")
    source_id = raw.get("id", idx)
    uid = f"{data_source}-{source_id}"

    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": user_content}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": solution_cleaned},
        "extra_info": {
            "index": idx,
            "source_id": source_id,
            "dataset_from": raw.get("dataset_from"),
            "problem": problem,
            "answer": answer,
        },
        "uid": uid,
    }


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_no}: {e}") from e
    if not rows:
        raise ValueError(f"No valid rows found in {path}")
    return rows


def _split_examples(examples: list[dict], test_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    if not (0.0 <= test_ratio < 1.0):
        raise ValueError(f"test_ratio must be in [0, 1), got {test_ratio}")

    total = len(examples)
    if total <= 1 or test_ratio == 0.0:
        train = deepcopy(examples)
        test = []
    else:
        indices = list(range(total))
        random.Random(seed).shuffle(indices)
        test_size = int(total * test_ratio)
        if test_size == 0:
            test_size = 1
        test_size = min(test_size, total - 1)
        test_idx = set(indices[:test_size])
        train = []
        test = []
        for i, ex in enumerate(examples):
            if i in test_idx:
                test.append(deepcopy(ex))
            else:
                train.append(deepcopy(ex))

    for ex in train:
        ex.setdefault("extra_info", {})
        ex["extra_info"]["split"] = "train"
    for ex in test:
        ex.setdefault("extra_info", {})
        ex["extra_info"]["split"] = "test"

    return train, test


def preprocess_jsonl_to_parquet(
    input_jsonl: str,
    output_dir: str,
    instruction: str = DEFAULT_INSTRUCTION,
    test_ratio: float = 0.0,
    seed: int = 42,
    write_test_parquet: bool = False,
) -> tuple[str, str | None]:
    """Convert JSONL to train/test parquet files for VERL RLHF training."""
    raw_rows = _load_jsonl(input_jsonl)
    normalized = [_normalize_example(raw=row, idx=i, instruction=instruction) for i, row in enumerate(raw_rows)]
    train_rows, test_rows = _split_examples(normalized, test_ratio=test_ratio, seed=seed)

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet") if (test_rows or write_test_parquet) else None

    train_dataset = datasets.Dataset.from_list(train_rows)
    train_dataset.to_parquet(train_path)

    if test_path is not None:
        if test_rows:
            test_dataset = datasets.Dataset.from_list(test_rows)
        else:
            # Keep the same schema when test split is empty.
            test_dataset = train_dataset.select([])
        test_dataset.to_parquet(test_path)

    with open(os.path.join(output_dir, "train_example.json"), "w", encoding="utf-8") as f:
        json.dump(train_rows[0], f, ensure_ascii=False, indent=2)
        f.write("\n")

    if test_rows:
        with open(os.path.join(output_dir, "test_example.json"), "w", encoding="utf-8") as f:
            json.dump(test_rows[0], f, ensure_ascii=False, indent=2)
            f.write("\n")

    return train_path, test_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--test_ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--write_test_parquet",
        action="store_true",
        help="Write test.parquet even when test split is empty.",
    )
    args = parser.parse_args()

    train_path, test_path = preprocess_jsonl_to_parquet(
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        instruction=args.instruction,
        test_ratio=args.test_ratio,
        seed=args.seed,
        write_test_parquet=args.write_test_parquet,
    )
    print(f"Saved train parquet: {train_path}")
    if test_path is not None:
        print(f"Saved test parquet:  {test_path}")


if __name__ == "__main__":
    main()
