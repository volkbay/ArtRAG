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
    rerank_blurb_words: int = 80
    context_word_budget: List[int] = field(
        default_factory=lambda: [300, 250, 200, 150, 100]
    )
    # BART summarizer
    bart_model_path: str = "./bin/pretrained/bart-large-cnn"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Where the general-flow lightrag.log is written (None -> working_dir, legacy).
    log_dir: str = None


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
    bart = _get("models.bart", None)
    if bart is not None:
        settings.bart_model_path = str(bart)
    log_dir = _get("log_dir", None)
    if log_dir is not None:
        settings.log_dir = str(log_dir)
