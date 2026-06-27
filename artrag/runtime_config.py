"""Runtime settings bridge for ArtRAG.

The REPRISE orchestration layer owns one OmegaConf config (``cfg/default.yaml``).
The artrag package can't depend on that file, so this module holds a small
process-wide ``settings`` object with the artrag-internal knobs that used to be
hard-coded constants (rerank budgets, summarizer model/device, vdb threshold,
the verbose switch). ``configure(cfg)`` is called once by the scripts to populate
it; modules read ``from .runtime_config import settings``.

Defaults below preserve the previous hard-coded behavior, so importing artrag
without calling ``configure`` keeps working.
"""

from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class Settings:
    verbose: bool = False
    # Retrieval / vdb
    cosine_threshold: float = 0.2
    # Reranking (prunning.py)
    rerank_strategy: str = "full"   # "full" (listwise) | "categorical" (pointwise graded)
    rerank_batch_size: int = 12     # entities per VLM call in categorical reranking
    rerank_score_threshold: float = 0.5  # categorical: keep nodes with final_score >= this (0-1)
    rerank_blurb_words: int = 80
    context_word_budget: List[int] = field(
        default_factory=lambda: [300, 250, 200, 150, 100]
    )
    rerank_max_attempts: int = 3
    # Summarizer (BART by default; InternVL3 path also supported)
    summarizer_model_path: str = "./bin/pretrained/bart-large-cnn"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Attention kernel for the InternVL3 LLM: "sdpa" (PyTorch memory-efficient,
    # default), "eager" (materializes the full heads x seq^2 score matrix —
    # OOM-prone at large top_k), or "flash_attention_2" (needs flash-attn built).
    attn_implementation: str = "sdpa"
    # Where the general-flow lightrag.log is written (None -> working_dir, legacy).
    log_dir: str = None
    # Diagnostic: log a VRAM snapshot at each model load / generation / rerank stage.
    log_gpu_memory: bool = False


settings = Settings()


def configure(cfg) -> None:
    """Populate the global ``settings`` from a REPRISE OmegaConf config.

    Accepts a partial config; only the recognized keys are applied. Safe to call
    more than once.
    """
    if cfg is None:
        return

    def _get(path, default):
        node = cfg
        for key in path.split("."):
            if node is None or key not in node:
                return default
            node = node[key]
        return node

    settings.verbose = bool(_get("verbose", settings.verbose))
    settings.cosine_threshold = float(
        _get("query.cosine_threshold", settings.cosine_threshold)
    )
    settings.rerank_blurb_words = int(
        _get("rerank.blurb_words", settings.rerank_blurb_words)
    )
    budget = _get("rerank.context_word_budget", None)
    if budget is not None:
        settings.context_word_budget = list(budget)
    settings.rerank_strategy = str(_get("rerank.strategy", settings.rerank_strategy))
    settings.rerank_batch_size = int(
        _get("rerank.batch_size", settings.rerank_batch_size)
    )
    settings.rerank_score_threshold = float(
        _get("rerank.score_threshold", settings.rerank_score_threshold)
    )
    settings.rerank_max_attempts = int(
        _get("rerank.max_attempts", settings.rerank_max_attempts)
    )
    summarizer = _get("models.summarizer", None)
    if summarizer is not None:
        settings.summarizer_model_path = str(summarizer)
    settings.attn_implementation = str(
        _get("models.attn_implementation", settings.attn_implementation)
    )
    log_dir = _get("log_dir", None)
    if log_dir is not None:
        settings.log_dir = str(log_dir)
    settings.log_gpu_memory = bool(_get("gpu.log_memory", settings.log_gpu_memory))
