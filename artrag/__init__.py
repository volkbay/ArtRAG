from .lightrag import LightRAG as LightRAG, QueryParam as QueryParam
from .runtime_config import configure as configure, settings as settings

# Import evaluation and inference utilities
from . import evaluation
from . import inference_utils

__version__ = "0.0.7"
__author__ = "Zirui Guo"
__url__ = "https://github.com/HKUDS/LightRAG"
