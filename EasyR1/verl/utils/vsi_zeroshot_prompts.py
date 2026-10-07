"""
GPD mixed RL: Student / Teacher conversation construction (use_vsi_zeroshot_prompts).

Selects system and user layout based on **question type** (choice / numeric) and **privileged variant**
(parquet ``extra_info.priv_variant``):
  - Student: always uses only Student RGB (``images``), no privilege.
  - Teacher: text_* injects ``priv_context_text``; image_* uses ``teacher_images`` (RGB+depth/sem/bev) + reference answer text.
  - pure_grpo: Teacher and Student share the same system prompt (no privilege block).
  - answer_only: Teacher receives only the ``<reference_answer>`` block (can be paired with data.teacher_prompt_answer_only).
  - text_routed_no_answer: Teacher receives only the ``<scene_context>`` routed 3D text, without ``<reference_answer>``.

User layout: ``[images...] + [optional priv text] + Question: {q_text}``
``q_text`` comes from parquet (already includes CoT / <answer> format instructions) and is not appended again.
"""

from __future__ import annotations

import os
import re
from typing import Any, Literal, Optional

AnswerKind = Literal["choice", "numeric"]
PrivVariant = Literal[
    "pure_grpo",
    "text_routed",
    "text_routed_no_answer",
    "text_full",
    "image_routed",
    "image_full",
    "answer_only",
]

CHOICE_FORMATS = frozenset({"mc", "select"})
NUMERIC_FORMATS = frozenset({"open", "fill"})
IMAGE_PRIV_VARIANTS = frozenset({"image_routed", "image_full"})
TEXT_PRIV_VARIANTS = frozenset({"text_routed", "text_full"})
TEXT_PRIV_NO_ANSWER_VARIANTS = frozenset({"text_routed_no_answer"})

# ── Student system ────────────────────────────────────────────────────

STUDENT_SYSTEM_CHOICE = (
    "You are a spatial reasoning assistant. "
    "The question is single-choice. "
    "Please answer based on the video frames."
)

STUDENT_SYSTEM_NUMERIC = (
    "You are a spatial reasoning assistant. "
    "Answer with a numerical value based on the video frames."
)

# ── Teacher system (text privilege: scene + optional reference_answer) ──────────

TEACHER_SYSTEM_TEXT_CHOICE = (
    "You are a spatial reasoning assistant. "
    "The question is single-choice. "
    "You receive the video frames, a <scene_context> block with structured 3D cues, and a <reference_answer> block. "
    "After understanding the <reference_answer>, please reason from the video frames and the structured 3D cues "
    "in the <scene_context> to solve the problem in your own way and derive the final answer."
)

TEACHER_SYSTEM_TEXT_NUMERIC = (
    "You are a spatial reasoning assistant. "
    "You receive the video frames, a <scene_context> block with structured 3D cues, and a <reference_answer> block. "
    "After understanding the <reference_answer>, please reason from the video frames and the structured 3D cues "
    "in the <scene_context> to solve the problem in your own way and derive the final numeric answer."
)

# ── Teacher system (text privilege: scene_context only, no reference_answer) ────

TEACHER_SYSTEM_TEXT_NO_ANSWER_CHOICE = (
    "You are a spatial reasoning assistant. "
    "The question is single-choice. "
    "You receive the video frames and a <scene_context> block with structured 3D cues. "
    "Please reason from the video frames and the structured 3D cues in the <scene_context> "
    "to solve the problem in your own way and derive the final answer."
)

TEACHER_SYSTEM_TEXT_NO_ANSWER_NUMERIC = (
    "You are a spatial reasoning assistant. "
    "You receive the video frames and a <scene_context> block with structured 3D cues. "
    "Please reason from the video frames and the structured 3D cues in the <scene_context> "
    "to solve the problem in your own way and derive the final numeric answer."
)

# ── Teacher system (image privilege: auxiliary 3D images + reference answer text) ────────────────

TEACHER_SYSTEM_IMAGE_CHOICE = (
    "You are a spatial reasoning assistant. "
    "The question is single-choice. "
    "You receive the video frames, auxiliary 3D images "
    "(such as colorized depth, instance segmentation, or bird's-eye view), and a <reference_answer> block. "
    "After understanding the <reference_answer>, please reason from the video frames and those auxiliary 3D images "
    "to solve the problem in your own way and derive the final answer."
)

TEACHER_SYSTEM_IMAGE_NUMERIC = (
    "You are a spatial reasoning assistant. "
    "You receive the video frames, auxiliary 3D images "
    "(such as colorized depth, instance segmentation, or bird's-eye view), and a <reference_answer> block. "
    "After understanding the <reference_answer>, please reason from the video frames and those auxiliary 3D images "
    "to solve the problem in your own way and derive the final numeric answer."
)

# ── Teacher system (reference answer block only, no 3D context) ────────────────────────────

TEACHER_SYSTEM_ANSWER_ONLY_CHOICE = (
    "You are a spatial reasoning assistant. "
    "The question is single-choice. "
    "You receive the video frames and a <reference_answer> block. "
    "After understanding the <reference_answer>, please reason from the video frames "
    "to solve the problem in your own way and derive the final answer."
)

TEACHER_SYSTEM_ANSWER_ONLY_NUMERIC = (
    "You are a spatial reasoning assistant. "
    "You receive the video frames and a <reference_answer> block. "
    "After understanding the <reference_answer>, please reason from the video frames "
    "to solve the problem in your own way and derive the final numeric answer."
)


def answer_kind_from_format(answer_format: str | None) -> AnswerKind:
    af = (answer_format or "mc").lower()
    if af in NUMERIC_FORMATS:
        return "numeric"
    return "choice"


def priv_variant_from_example(example: dict[str, Any]) -> str:
    extra = example.get("extra_info") or {}
    if not isinstance(extra, dict):
        extra = {}
    v = extra.get("priv_variant") or extra.get("priv_context_mode") or "pure_grpo"
    return str(v)


def meta_from_example(example: dict[str, Any]) -> tuple[AnswerKind, str]:
    extra = example.get("extra_info") or {}
    if not isinstance(extra, dict):
        extra = {}
    if example.get("vsi_answer_kind"):
        return str(example["vsi_answer_kind"]), priv_variant_from_example(example)  # type: ignore[return-value]
    af = extra.get("answer_format")
    return answer_kind_from_format(str(af) if af is not None else None), priv_variant_from_example(example)


def resolve_image_path(path: str, image_dir: Optional[str] = None) -> str:
    if image_dir and path and not os.path.isabs(path):
        return os.path.join(image_dir, path)
    return path


def parse_image_paths(images_field: Any, image_dir: Optional[str] = None) -> list[str]:
    if not images_field:
        return []
    out: list[str] = []
    for im in images_field:
        if isinstance(im, dict):
            p = im.get("path") or im.get("image")
            if p:
                out.append(resolve_image_path(str(p), image_dir))
        elif im:
            out.append(resolve_image_path(str(im), image_dir))
    return out


def extract_vsi_q_text(example: dict[str, Any], prompt_key: str) -> str:
    """Extract the question text from the conversations or prompt field, stripping <image> placeholders."""
    if "conversations" in example and example["conversations"]:
        raw = example["conversations"][0].get("value", "")
        return re.sub(r"(<image>\s*)+", "", raw).strip()

    pv = example.get(prompt_key)
    if isinstance(pv, list):
        for m in pv:
            if m.get("role") != "user":
                continue
            c = m.get("content")
            if isinstance(c, str):
                return re.sub(r"(<image>\s*)+", "", c).strip()
            if isinstance(c, list):
                parts: list[str] = []
                for p in c:
                    if isinstance(p, dict) and p.get("type") == "text":
                        parts.append(p.get("text", ""))
                return re.sub(r"(<image>\s*)+", "", "".join(parts)).strip()

    if isinstance(pv, str):
        return re.sub(r"(<image>\s*)+", "", pv).strip()

    raise ValueError(
        "use_vsi_zeroshot_prompts: need `conversations` or parsable `prompt` to extract q_text."
    )


def _student_system(answer_kind: AnswerKind) -> str:
    return STUDENT_SYSTEM_NUMERIC if answer_kind == "numeric" else STUDENT_SYSTEM_CHOICE


def _teacher_system(
    answer_kind: AnswerKind,
    priv_variant: str,
    *,
    teacher_prompt_answer_only: bool = False,
) -> str:
    pv = priv_variant or "pure_grpo"
    if pv == "pure_grpo":
        return _student_system(answer_kind)
    if teacher_prompt_answer_only or pv == "answer_only":
        return (
            TEACHER_SYSTEM_ANSWER_ONLY_NUMERIC
            if answer_kind == "numeric"
            else TEACHER_SYSTEM_ANSWER_ONLY_CHOICE
        )
    if pv in IMAGE_PRIV_VARIANTS:
        return (
            TEACHER_SYSTEM_IMAGE_NUMERIC
            if answer_kind == "numeric"
            else TEACHER_SYSTEM_IMAGE_CHOICE
        )
    if pv in TEXT_PRIV_NO_ANSWER_VARIANTS:
        return (
            TEACHER_SYSTEM_TEXT_NO_ANSWER_NUMERIC
            if answer_kind == "numeric"
            else TEACHER_SYSTEM_TEXT_NO_ANSWER_CHOICE
        )
    if pv in TEXT_PRIV_VARIANTS:
        return (
            TEACHER_SYSTEM_TEXT_NUMERIC
            if answer_kind == "numeric"
            else TEACHER_SYSTEM_TEXT_CHOICE
        )
    return _student_system(answer_kind)


def _build_vsi_user_content_cot(
    n_images: int,
    q_text: str,
    *,
    priv_context: str | None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "image"} for _ in range(n_images)]
    if priv_context is not None:
        p = (priv_context or "").strip()
        if p and p != "<scene_context>\n</scene_context>":
            content.append({"type": "text", "text": "\n" + p + "\n"})
    content.append({"type": "text", "text": "\nQuestion: " + q_text + "\n"})
    return content


def build_student_message_list(
    n_images: int,
    q_text: str,
    *,
    answer_kind: AnswerKind = "choice",
) -> list[dict[str, Any]]:
    content = _build_vsi_user_content_cot(n_images, q_text, priv_context=None)
    return [
        {"role": "system", "content": _student_system(answer_kind)},
        {"role": "user", "content": content},
    ]


def build_teacher_message_list(
    n_images: int,
    q_text: str,
    priv_context: str,
    *,
    answer_kind: AnswerKind = "choice",
    priv_variant: str = "pure_grpo",
    teacher_prompt_answer_only: bool = False,
) -> list[dict[str, Any]]:
    content = _build_vsi_user_content_cot(n_images, q_text, priv_context=priv_context)
    sys_txt = _teacher_system(
        answer_kind,
        priv_variant,
        teacher_prompt_answer_only=teacher_prompt_answer_only,
    )
    return [
        {"role": "system", "content": sys_txt},
        {"role": "user", "content": content},
    ]


def resolve_teacher_image_paths(
    example: dict[str, Any],
    priv_variant: str,
    student_image_paths: list[str],
    teacher_images_key: str = "teacher_images",
    image_dir: Optional[str] = None,
) -> list[str]:
    """Use teacher_images for image_* variants; fall back to student image paths otherwise."""
    if priv_variant not in IMAGE_PRIV_VARIANTS:
        return list(student_image_paths)
    tpaths = parse_image_paths(example.get(teacher_images_key), image_dir=image_dir)
    if tpaths:
        return tpaths
    return list(student_image_paths)
