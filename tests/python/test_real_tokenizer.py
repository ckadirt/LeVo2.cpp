"""Opt-in conformance test against the pinned Transformers Qwen2 tokenizer."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


CASES = [
    "<|im_start|>[verse]Hello world.",
    "123 4567 ９８７",
    "it's WE'RE can't she'll",
    "!!!??? …—♪\n\n",
    "  spaces\tand\r\nlines  ",
    "中文歌词，月亮与海。",
    "Español déjà vu; Ελληνικά",
    "Привет, мир! العربية",
    "한글 노래 🎵🌙",
    "e\u0301 versus é",
]


def test_cpp_matches_pinned_qwen2() -> None:
    tokenizer_dir_value = os.environ.get("LEVO_TOKENIZER_DIR")
    tool_value = os.environ.get("LEVO_TOKENIZER_PARITY_TOOL")
    if not tokenizer_dir_value or not tool_value:
        pytest.skip("set LEVO_TOKENIZER_DIR and LEVO_TOKENIZER_PARITY_TOOL for real tokenizer parity")

    tokenizer_dir = Path(tokenizer_dir_value)
    tool = Path(tool_value)
    if not tokenizer_dir.is_dir() or not tool.is_file():
        pytest.fail("real tokenizer parity paths do not exist")

    transformers = pytest.importorskip("transformers")
    official = transformers.Qwen2Tokenizer.from_pretrained(
        tokenizer_dir, local_files_only=True
    )
    official.add_tokens(
        [
            "[verse]", "[chorus]", "[bridge]", "[intro-short]", "[intro-medium]",
            "[intro-long]", "[outro-short]", "[outro-medium]", "[outro-long]",
            "[inst-short]", "[inst-medium]", "[inst-long]", "[silence]",
        ],
        special_tokens=True,
    )

    for text in CASES:
        expected = official.encode(text, add_special_tokens=False)
        completed = subprocess.run(
            [str(tool), str(tokenizer_dir), text],
            check=True,
            capture_output=True,
            text=True,
        )
        actual = [int(value) for value in completed.stdout.split()]
        assert actual == expected, text
