
"""
Preprocess open-r1/DAPO-Math-17k-Processed (train) and local AIME jsonl (test)
into VERL RLHF parquet format. All settings are hardcoded.
"""

import json
import os

import datasets

INSTRUCTION = "\nPlease think step by step and place your final answer inside \\boxed{}"
TRAIN_DATASET = "open-r1/DAPO-Math-17k-Processed"
TRAIN_SPLIT = "train"
TEST_JSONL = "/Users/shengke/Downloads/aime25.jsonl"
SAVE_DIR = "./dapo_math17k_aime25_parquet"

TRAIN_DATA_SOURCE = TRAIN_DATASET
TEST_DATA_SOURCE = "aime25"
ABILITY = "math"


def _ensure_extra_info(extra_info, split, idx, question=None, answer=None):
    if not isinstance(extra_info, dict):
        extra_info = {}
    extra_info.setdefault("split", split)
    extra_info.setdefault("index", idx)
    if question is not None:
        extra_info.setdefault("question", question)
    if answer is not None:
        extra_info.setdefault("answer", answer)
    return extra_info


def _map_train(example, idx):
    # If dataset already follows VERL schema, just add extra_info defaults.

    # Minimal fallback: assume problem/answer fields exist.
    question = example.get("prompt")
    answer = example.get("solution")
    if question is None or answer is None:
        raise ValueError("Train example missing prompt/reward_model and problem/answer fields.")
    prompt = [{"role": "user", "content": question+INSTRUCTION}]
    return {
        "data_source": TRAIN_DATA_SOURCE,
        "prompt": prompt,
        "ability": ABILITY,
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": _ensure_extra_info({}, "train", idx, question, answer),
    }


def _map_test(example, idx):
    question = example["problem"]
    answer = example["answer"]
    prompt = [{"role": "user", "content": question+INSTRUCTION}]
    return {
        "data_source": TEST_DATA_SOURCE,
        "prompt": prompt,
        "ability": ABILITY,
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": _ensure_extra_info({}, "test", idx, question, answer),
    }


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    train_dataset = datasets.load_dataset(TRAIN_DATASET,"en", split=TRAIN_SPLIT)
    test_dataset = datasets.load_dataset("json", data_files=TEST_JSONL, split="train")

    train_dataset = train_dataset.map(function=_map_train, with_indices=True)
    test_dataset = test_dataset.map(function=_map_test, with_indices=True)

    train_dataset.to_parquet(os.path.join(SAVE_DIR, "train.parquet"))
    test_dataset.to_parquet(os.path.join(SAVE_DIR, "test.parquet"))

    with open(os.path.join(SAVE_DIR, "train_example.json"), "w") as f:
        json.dump(train_dataset[0], f, indent=2)
    with open(os.path.join(SAVE_DIR, "test_example.json"), "w") as f:
        json.dump(test_dataset[0], f, indent=2)


if __name__ == "__main__":
    main()
