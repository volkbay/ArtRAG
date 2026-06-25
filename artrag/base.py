from dataclasses import dataclass, field
from typing import TypedDict, Union, Literal, Generic, TypeVar

import numpy as np

from .utils import EmbeddingFunc

TextChunkSchema = TypedDict(
    "TextChunkSchema",
    {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int},
)

T = TypeVar("T")


@dataclass
class QueryParam:
    mode: Literal["local", "global", "hybrid", "naive"] = "global"
    only_need_context: bool = False
    # response_type: str = "Keep your generated description strictly under 15 words."
    response_type: str = "Include context e.g. Cultural, Historical and theme in your generated description. Keep your generated description strictly under 30 words. "
    top_k: int = 5              # entities from the vdb (initial retrieval) + naive chunks
    top_k_expansion: int = 5    # one-hop edges kept when expanding retrieved nodes
    top_k_rerank: int = 5       # entities kept after VLM/degree reranking (final context)
    max_token_for_text_unit: int = 1000
    max_token_for_global_context: int = 1000
    max_token_for_local_context: int = 1000
    vlm_weight: float = 0.5  # Weight for VLM scores in reranking (0-1)
    data_type: str = "SemArtv2"  # Dataset type (SemArtv2, Artpedia)
    shot_number: int = 1  # Number of few-shot examples (0, 1, 2, 3)
    fewshot_type: str = "SM_fewshot"  # Type of few-shot: "SM_fewshot" or "MM_fewshot"


@dataclass
class StorageNameSpace:
    namespace: str
    global_config: dict

    async def index_done_callback(self):
        """commit the storage operations after indexing"""
        pass
    async def initialize(self):
        """Initialize the storage"""
        pass

    async def query_done_callback(self):
        """commit the storage operations after querying"""
        pass


@dataclass
class BaseVectorStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc
    meta_fields: set = field(default_factory=set)

    async def query(self, query: str, top_k: int) -> list[dict]:
        raise NotImplementedError

    async def upsert(self, data: dict[str, dict]):
        """Use 'content' field from value for embedding, use key as id.
        If embedding_func is None, use 'embedding' field from value
        """
        raise NotImplementedError
    
    async def delete_entity(self, entity_name: str) -> None:
        """Delete a single entity by its name."""

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete relations for a given entity."""


@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    async def all_keys(self) -> list[str]:
        raise NotImplementedError

    async def get_by_id(self, id: str) -> Union[T, None]:
        raise NotImplementedError

    async def get_by_ids(
        self, ids: list[str], fields: Union[set[str], None] = None
    ) -> list[Union[T, None]]:
        raise NotImplementedError

    async def filter_keys(self, data: list[str]) -> set[str]:
        """return un-exist keys"""
        raise NotImplementedError

    async def upsert(self, data: dict[str, T]):
        raise NotImplementedError

    async def drop(self):
        raise NotImplementedError


@dataclass
class BaseGraphStorage(StorageNameSpace):
    async def has_node(self, node_id: str) -> bool:
        raise NotImplementedError

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        raise NotImplementedError

    async def node_degree(self, node_id: str) -> int:
        raise NotImplementedError

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        raise NotImplementedError

    async def get_node(self, node_id: str) -> Union[dict, None]:
        raise NotImplementedError

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        raise NotImplementedError

    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        raise NotImplementedError

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        raise NotImplementedError

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        raise NotImplementedError

    async def clustering(self, algorithm: str):
        raise NotImplementedError

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        raise NotImplementedError("Node embedding is not used in lightrag.")
