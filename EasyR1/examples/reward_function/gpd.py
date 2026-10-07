"""
GPD spatial reasoning reward function (EasyR1 batch format).

Aligns with VSIBench / SPAR-Bench evaluation logic:
  - Multiple-choice (mc / select): exact_match (letter fuzzy matching)
  - Numeric fill-in (open / fill): MRA mean_relative_accuracy (.5:.95:.05)

Answer format: <think>...</think><answer>...</answer>
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

VALID_OPTIONS = frozenset({"A", "B", "C", "D", "E"})

# Default MRA configuration for VSIBench / SPAR numeric answers
MRA_START = 0.5
MRA_END = 0.95
MRA_INTERVAL = 0.05


def clean_answer_text(text: str) -> str:
    """Extract the last <answer> block and normalize whitespace."""
    answer_matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if answer_matches:
        text = answer_matches[-1]
    for char in ("\n", "\r"):
        text = re.sub(r"(?<=\s)" + re.escape(char), "", text)
        text = re.sub(r"(?<!\s)" + re.escape(char), " ", text)
    return text.strip().rstrip(".").lower()


def fuzzy_matching(pred: str) -> str:
    """Match MCA answer from the first-token letter (vsi_utils.fuzzy_matching, extended to E)."""
    pred = pred.strip()
    m = re.search(r"^([A-E])\.?", pred.split(" ")[0].strip(), re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return pred.strip()


def fuzzy_matching_num(pred: str) -> str:
    """Extract a numeric value from the answer (vsi_utils.fuzzy_matching_num)."""
    pred = pred.strip().lower()
    number_words = {
        "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
        "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
        "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20",
        "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
        "eighty": "80", "ninety": "90", "zero": "0", "a": "1", "an": "1",
    }
    for word, digit in number_words.items():
        if re.search(r"\b" + word + r"\b", pred):
            return digit
    m = re.search(r"(\d+(?:\.\d+)?)", pred)
    if m:
        return m.group(1)
    return "None"


def to_float(val: str | float | None) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def abs_dist_norm(pred: float, target: float) -> float:
    """Normalized absolute distance (spar_utils.abs_dist_norm): uses absolute error when target=0."""
    if target == 0.0:
        return abs(pred - target)
    return abs((pred - target) / target)


def mean_relative_accuracy(
    pred: float | None,
    target: float | None,
    start: float = MRA_START,
    end: float = MRA_END,
    interval: float = MRA_INTERVAL,
) -> float:
    """VSIBench MRA: thresholds from 0.5 to 0.95 in steps of 0.05 (vsi_utils.mean_relative_accuracy)."""
    if pred is None or target is None:
        return 0.0
    num_pts = int((end - start) / interval + 2)
    conf_intervs = np.linspace(start, end, num_pts)
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return float(accuracy.mean())


def exact_match(pred: str, target: str) -> float:
    return 1.0 if pred.lower() == target.lower() else 0.0


def _is_choice_ground_truth(gt: str) -> bool:
    s = gt.strip()
    return len(s) == 1 and s.isalpha()


def _extract_answer_content(response: str) -> str:
    """Extract the text inside <answer> from the full response; falls back to the entire response if no tag."""
    cleaned = clean_answer_text(response)
    if cleaned:
        return cleaned
    return response.strip()


def extract_choice_answer(response: str) -> str:
    content = _extract_answer_content(response)
    letter = fuzzy_matching(content)
    if letter.upper() in VALID_OPTIONS:
        return letter.upper()
    # fallback: search for a letter option in the full response
    m = re.search(r"<answer>\s*([A-E])\s*</answer>", response, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    candidates = re.findall(r"\b([A-E])\b", response.upper())
    return candidates[-1] if candidates else ""


def extract_numeric_for_mra(response: str) -> float | None:
    content = _extract_answer_content(response)
    num_str = fuzzy_matching_num(content)
    if num_str == "None":
        return None
    return to_float(num_str)


def extract_final_answer(response: str) -> str:
    """Legacy interface compatibility."""
    letter = extract_choice_answer(response)
    if letter:
        return letter
    num = extract_numeric_for_mra(response)
    return str(num) if num is not None else ""


def mca_accuracy_reward(response: str, ground_truth: str) -> float:
    pred_raw = _extract_answer_content(response)
    gt_raw = clean_answer_text(f"<answer>{ground_truth}</answer>")
    if not gt_raw:
        gt_raw = ground_truth.strip().lower()
    pred = fuzzy_matching(pred_raw.replace("answer:", ""))
    target = fuzzy_matching(gt_raw.replace("answer:", ""))
    return exact_match(pred, target)


def na_accuracy_reward(response: str, ground_truth: str) -> float:
    pred = extract_numeric_for_mra(response)
    target = to_float(fuzzy_matching_num(clean_answer_text(f"<answer>{ground_truth}</answer>") or ground_truth))
    return mean_relative_accuracy(pred, target)


def accuracy_reward(response: str, ground_truth: str) -> float:
    gt = ground_truth.strip()
    if _is_choice_ground_truth(gt):
        return mca_accuracy_reward(response, gt)
    return na_accuracy_reward(response, gt)


def format_reward(response: str, *, choice: bool = True) -> float:
    has_think = bool(re.search(r"<think>.*?</think>", response, re.DOTALL))
    if choice:
        has_answer_tag = bool(re.search(r"<answer>\s*[A-E]\s*</answer>", response, re.IGNORECASE))
        has_answer = extract_choice_answer(response) in VALID_OPTIONS
    else:
        has_answer_tag = bool(re.search(r"<answer>\s*[^<]+\s*</answer>", response, re.IGNORECASE))
        has_answer = extract_numeric_for_mra(response) is not None

    if has_think and has_answer_tag:
        return 1.0
    if has_answer:
        return 0.5
    return 0.0


def length_penalty(response_length: int, length_target: int, length_pen: float) -> float:
    if length_pen <= 0.0 or length_target <= 0:
        return 0.0
    if response_length >= length_target:
        return 0.0
    return -float(length_pen) * (1.0 - float(response_length) / float(length_target))


def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.1,
    length_target: int = 0,
    length_pen: float = 0.0,
) -> list[dict[str, float]]:
    """
    EasyR1 batch reward interface.

    Accuracy for numeric questions is MRA in [0, 1] (non-binary), consistent with VSIBench-style GRPO.
    """
    scores = []
    for item in reward_inputs:
        response = item.get("response", "")
        gt = item.get("ground_truth", "")
        if isinstance(gt, dict):
            gt = gt.get("ground_truth", "")
        gt_str = str(gt).strip()
        resp_len = int(item.get("response_length", 0) or 0)

        is_choice = _is_choice_ground_truth(gt_str)
        acc = accuracy_reward(response, gt_str)
        fmt = format_reward(response, choice=is_choice)
        len_pen = length_penalty(resp_len, length_target, length_pen)
        overall = (1.0 - format_weight) * acc + format_weight * fmt + len_pen

        row: dict[str, float] = {
            "overall": overall,
            "accuracy": acc,
            "format": fmt,
            "length_penalty": len_pen,
            "response_length": float(resp_len),
        }
        if not is_choice:
            row["mra"] = acc
        scores.append(row)
    return scores
