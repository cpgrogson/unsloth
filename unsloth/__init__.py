# Copyright 2023-present, the Unsloth team.
# Licensed under the Apache License, Version 2.0

"""
Unsloth - Fast LLM finetuning with minimal memory usage.

This package provides optimized training utilities for large language models,
including efficient LoRA/QLoRA implementations and custom CUDA kernels.
"""

__version__ = "2024.1.0"
__author__ = "Unsloth Team"
__license__ = "Apache 2.0"

import sys
import os

# Minimum Python version check
if sys.version_info < (3, 8):
    raise RuntimeError(
        "Unsloth requires Python 3.8 or higher. "
        f"You are running Python {sys.version_info.major}.{sys.version_info.minor}."
    )

# Check for required dependencies before importing submodules
def _check_dependencies():
    """Verify that core dependencies are available."""
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        import transformers
    except ImportError:
        missing.append("transformers")

    try:
        import peft
    except ImportError:
        missing.append("peft")

    if missing:
        raise ImportError(
            f"Unsloth requires the following packages: {', '.join(missing)}. "
            "Install them with: pip install " + " ".join(missing)
        )


_check_dependencies()

# Core imports exposed at the package level
from unsloth.models import (
    FastLanguageModel,
)
from unsloth.trainer import (
    UnslothTrainer,
    UnslothTrainingArguments,
)
from unsloth.chat_templates import get_chat_template

__all__ = [
    "FastLanguageModel",
    "UnslothTrainer",
    "UnslothTrainingArguments",
    "get_chat_template",
    "__version__",
]

# Optional: display a startup banner in interactive environments
def _print_banner():
    """Print a startup banner when running in an interactive session."""
    import torch
    cuda_available = torch.cuda.is_available()
    device_info = torch.cuda.get_device_name(0) if cuda_available else "CPU only"
    # Also show available VRAM when a GPU is present, useful for planning batch sizes
    if cuda_available:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        device_info = f"{device_info} ({vram_gb:.1f} GB VRAM)"
    print(
        f"🦥 Unsloth v{__version__} | "
        f"Device: {device_info} | "
        f"Torch: {torch.__version__}"
    )


if os.environ.get("UNSLOTH_SILENT", "0") != "1":
    try:
        _print_banner()
    except Exception:
        pass  # Never fail on banner printing
