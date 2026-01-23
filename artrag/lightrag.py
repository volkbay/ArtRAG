import asyncio
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Type, cast, Any, Dict, Union, List, Optional
import pdb
from .prompt_art import GRAPH_FIELD_SEP, PROMPTS

from .llm import (
    gpt_4o_mini_complete,
    openai_embedding,
)
from .operate import (
    chunking_by_token_size,
    extract_entities,
    local_query,
    # global_query,
    # hybrid_query,
    naive_query,
    no_rag,
)

from .storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)
from .utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    logger,
    set_logger,
    encode_image_to_base64,
    validate_image_file,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.info("Creating a new event loop in a sub-thread.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


@dataclass
class LightRAG:
    working_dir: str = field(
        default_factory=lambda: f"./lightrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )

    # text chunking
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o-mini"

    # entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # embedding_func: EmbeddingFunc = field(default_factory=lambda:hf_embedding)
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16

    # LLM
    llm_model_func: callable = gpt_4o_mini_complete  # hf_model_complete#
    # llm_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"  #'meta-llama/Llama-3.2-1B'#'google/gemma-2-2b-it'
    llm_model_max_token_size: int = 32768
    llm_model_max_async: int = 16
    llm_model_kwargs: dict = field(default_factory=dict)
    
    # Vision model (optional, for multimodal queries)
    vision_model_func: Optional[callable] = field(default=None)

    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage
    enable_llm_cache: bool = True

    # extension
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    def __post_init__(self):
        log_file = os.path.join(self.working_dir, "lightrag.log")
        set_logger(log_file)
        logger.info(f"Logger initialized for working directory: {self.working_dir}")

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"LightRAG init with param:\n  {_print_config}\n")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config=asdict(self)
        )

        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self)
        )

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=asdict(self)
            )
            if self.enable_llm_cache
            else None
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=asdict(self)
        )

        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )

        self.entities_vdb = self.vector_db_storage_cls(
            namespace="entities",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},
        )
        self.relationships_vdb = self.vector_db_storage_cls(
            namespace="relationships",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id"},
        )
        self.chunks_vdb = self.vector_db_storage_cls(
            namespace="chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )

        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
            partial(self.llm_model_func,
                     hashing_kv=self.llm_response_cache,
                     **self.llm_model_kwargs,)
        )

    def insert(self, string_or_strings):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    async def ainsert(self, string_or_strings):
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]

            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning("All docs are already in the storage")
                return
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            inserting_chunks = {}
            for doc_key, doc in new_docs.items():
                chunks = {
                    compute_mdhash_id(dp["content"], prefix="chunk-"): {
                        **dp,
                        "full_doc_id": doc_key,
                    }
                    for dp in chunking_by_token_size(
                        doc["content"],
                        overlap_token_size=self.chunk_overlap_token_size,
                        max_token_size=self.chunk_token_size,
                        tiktoken_model=self.tiktoken_model_name,
                    )
                }
                inserting_chunks.update(chunks)
            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")

            await self.chunks_vdb.upsert(inserting_chunks)

            logger.info("[Entity Extraction]...")
            maybe_new_kg = await extract_entities(
                inserting_chunks,
                knwoledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                relationships_vdb=self.relationships_vdb,
                global_config=asdict(self),
            )
            if maybe_new_kg is None:
                logger.warning("No new entities and relationships found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg

            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            await self._insert_done()

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.relationships_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def query(self, query: str, data_type="SemArtv2", shot_number=1,fewshot_type="SM_fewshot", param: QueryParam = QueryParam(), vlm_weight=0.5):
        loop = always_get_an_event_loop()
        param.data_type = data_type
        param.fewshot_type = fewshot_type
        param.shot_number = shot_number
        param.vlm_weight = vlm_weight
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):

        if param.mode == "local":
            response, beforererank_context, rerank_context = await local_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.text_chunks,
                param,
                asdict(self),
            )
        elif param.mode == "naive":
            response = await naive_query(
                query,
                self.chunks_vdb,
                self.text_chunks,
                param,
                asdict(self),
            )
            beforererank_context = None
            rerank_context = None
        elif param.mode == "no-rag":
            response = await no_rag(
                query,
                param,
                asdict(self),
            )
            beforererank_context = None
            rerank_context = None
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response, beforererank_context, rerank_context

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    async def aquery_with_agentic_reasoning(
        self,
        query: str,
        multimodal_content: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        mode: str = "local",
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Agentic multimodal query with planning and custom final generation.

        Flow:
        1) Generate retrieval + generation plans from multimodal query.
        2) Retrieve context using LightRAG (only_need_context=True).
        3) Build final prompt with generation plan and call LLM/VLM.

        Args:
            query: User query text
            multimodal_content: Optional list of multimodal content, each element contains:
                - type: Content type ("image", "table", "equation", etc.)
                - Other fields depend on type (e.g., img_path, table_data, latex, etc.)
            metadata: Optional metadata dictionary
            mode: Query mode ("local", "naive", "no-rag")
            system_prompt: Optional system prompt
            **kwargs: Other query parameters

        Returns:
            str: Query result
        """
        if not self.llm_model_func:
            raise ValueError(
                "llm_model_func is required for agentic reasoning. Please provide it when initializing LightRAG."
            )

        multimodal_content = multimodal_content or []
        metadata = metadata or {}

        logger.info(f"Executing agentic query: {query[:100]}...")
        logger.info(f"Query mode: {mode}")

        # Build planning prompt
        multimodal_summary = self._summarize_multimodal_content_for_planner(
            multimodal_content
        )
        plan_prompt = PROMPTS["AGENTIC_PLAN_PROMPT"].format(
            query=query,
            metadata=json.dumps(metadata, ensure_ascii=True),
            multimodal_summary=multimodal_summary,
        )

        # Generate plans (prefer VLM if images are available)
        plan_response = await self._call_agentic_planner(
            plan_prompt, multimodal_content
        )
        plan_data = self._parse_agentic_plan_response(plan_response)

        retrieval_plan = plan_data.get("retrieval_plan", {})
        retrieval_query = (
            plan_data.get("retrieval_query")
            or retrieval_plan.get("retrieval_query")
            or ""
        )
        missing_info = retrieval_plan.get("missing_info", [])
        generation_plan = plan_data.get("generation_plan", [])

        if not retrieval_query:
            retrieval_query = query
        
        # Retrieve context from LightRAG
        # Use naive mode for context retrieval as it properly handles only_need_context
        query_param = QueryParam(mode="naive", only_need_context=True, **kwargs)
        retrieved_context = await naive_query(
            retrieval_query,
            self.chunks_vdb,
            self.text_chunks,
            query_param,
            asdict(self),
        )

        if not isinstance(retrieved_context, str):
            retrieved_context = json.dumps(retrieved_context, ensure_ascii=True)

        # Build final answer prompt
        final_prompt = PROMPTS["AGENTIC_FINAL_ANSWER"].format(
            query=query,
            metadata=json.dumps(metadata, ensure_ascii=True),
            retrieved_context=retrieved_context,
            generation_plan=json.dumps(generation_plan, ensure_ascii=True),
        )

        # Include missing info in the final prompt if available
        if missing_info:
            final_prompt += (
                "\nMissing information considered for retrieval:\n"
                + json.dumps(missing_info, ensure_ascii=True)
                + "\n"
            )

        # Generate final answer using VLM if images are available
        if self._has_valid_images(multimodal_content) and self.vision_model_func:
            messages = self._build_vlm_messages_for_agentic(
                final_prompt,
                multimodal_content,
                system_prompt or PROMPTS["AGENTIC_FINAL_SYSTEM"],
            )
            result = await self.vision_model_func("", messages=messages)
        else:
            result = await self.llm_model_func(
                final_prompt, system_prompt=system_prompt or PROMPTS["AGENTIC_FINAL_SYSTEM"]
            )

        logger.info("Agentic query completed")
        return result

    def _summarize_multimodal_content_for_planner(
        self, multimodal_content: List[Dict[str, Any]]
    ) -> str:
        """
        Explain the multimodal content for the planner
        """
        if not multimodal_content:
            return "None"

        summaries = []
        for item in multimodal_content:
            content_type = item.get("type", "unknown")
            if content_type == "image":
                image_path = item.get("img_path") or item.get("image_path")
                captions = item.get("image_caption") or item.get("img_caption") or []
                footnotes = item.get("image_footnote") or item.get("img_footnote") or []
                summaries.append(
                    f"- image: path={image_path}, captions={captions}, footnotes={footnotes}"
                )
            elif content_type == "table":
                table_caption = item.get("table_caption", "")
                table_data = item.get("table_data", "")
                summaries.append(
                    f"- table: caption={table_caption}, data={str(table_data)[:200]}"
                )
            elif content_type == "equation":
                latex = item.get("latex", "")
                summaries.append(f"- equation: latex={latex}")
            else:
                summaries.append(f"- {content_type}: {str(item)[:200]}")

        return "\n".join(summaries)

    async def _call_agentic_planner(
        self, plan_prompt: str, multimodal_content: List[Dict[str, Any]]
    ) -> str:
        if self._has_valid_images(multimodal_content) and self.vision_model_func:
            messages = self._build_vlm_messages_for_agentic(
                plan_prompt, multimodal_content, PROMPTS["AGENTIC_PLAN_SYSTEM"]
            )
            return await self.vision_model_func("", messages=messages)

        return await self.llm_model_func(
            plan_prompt, system_prompt=PROMPTS["AGENTIC_PLAN_SYSTEM"]
        )

    def _has_valid_images(self, multimodal_content: List[Dict[str, Any]]) -> bool:
        for item in multimodal_content or []:
            if item.get("type") == "image":
                image_path = item.get("img_path") or item.get("image_path")
                if image_path and validate_image_file(image_path):
                    return True
        return False

    def _build_vlm_messages_for_agentic(
        self,
        prompt: str,
        multimodal_content: List[Dict[str, Any]],
        system_prompt: str,
    ) -> List[Dict]:
        content_parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        image_index = 1

        for item in multimodal_content:
            if item.get("type") != "image":
                continue

            image_path = item.get("img_path") or item.get("image_path")
            if not image_path or not validate_image_file(image_path):
                continue

            image_base64 = encode_image_to_base64(image_path)
            if not image_base64:
                continue

            content_parts.append(
                {"type": "text", "text": f"\n[Image {image_index}]\n"}
            )
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                }
            )
            image_index += 1

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts},
        ]

    def _parse_agentic_plan_response(self, response: Any) -> Dict[str, Any]:
        if response is None:
            return {}

        response_text = response if isinstance(response, str) else str(response)
        cleaned = response_text.strip()

        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9]*\n", "", cleaned).strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]

        try:
            return json.loads(cleaned)
        except Exception as e:
            logger.debug(f"Failed to parse agentic plan JSON: {e}")
            return {}

    def query_with_agentic_reasoning(
        self,
        query: str,
        multimodal_content: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        mode: str = "local",
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Synchronous version of agentic multimodal query with planning.
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aquery_with_agentic_reasoning(
                query,
                multimodal_content=multimodal_content,
                metadata=metadata,
                mode=mode,
                system_prompt=system_prompt,
                **kwargs,
            )
        )

    def delete_by_entity(self, entity_name: str) -> None:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name))

    async def adelete_by_entity(self, entity_name: str) -> None:
        try:
            await self.entities_vdb.delete_entity(entity_name)
            await self.relationships_vdb.delete_entity_relation(entity_name)
            await self.chunk_entity_relation_graph.delete_node(entity_name)

            logger.info(
                f"Entity '{entity_name}' and its relationships have been deleted."
            )
            await self._delete_by_entity_done()
        except Exception as e:
            logger.error(f"Error while deleting entity '{entity_name}': {e}")

    async def _delete_by_entity_done(self) -> None:
        await asyncio.gather(
            *[
                cast(StorageNameSpace, storage_inst).index_done_callback()
                for storage_inst in [  # type: ignore
                    self.entities_vdb,
                    self.relationships_vdb,
                    self.chunk_entity_relation_graph,
                ]
            ]
        )

    async def amerge_entities(
        self,
        source_entities: list[str],
        target_entity: str,
        merge_strategy: dict[str, str] = None,
        target_entity_data: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """Asynchronously merge multiple entities into one entity.

        Merges multiple source entities into a target entity, handling all relationships,
        and updating both the knowledge graph and vector database.

        Args:
            source_entities: List of source entity names to merge
            target_entity: Name of the target entity after merging
            merge_strategy: Merge strategy configuration, e.g. {"description": "concatenate", "entity_type": "keep_first"}
                Supported strategies:
                - "concatenate": Concatenate all values (for text fields)
                - "keep_first": Keep the first non-empty value
                - "keep_last": Keep the last non-empty value
                - "join_unique": Join all unique values (for fields separated by delimiter)
            target_entity_data: Dictionary of specific values to set for the target entity,
                overriding any merged values, e.g. {"description": "custom description", "entity_type": "PERSON"}

        Returns:
            Dictionary containing the merged entity information
        """
        try:
            # Default merge strategy
            default_strategy = {
                "description": "concatenate",
                "entity_type": "keep_first",
                "source_id": "join_unique",
            }

            merge_strategy = (
                default_strategy
                if merge_strategy is None
                else {**default_strategy, **merge_strategy}
            )
            target_entity_data = (
                {} if target_entity_data is None else target_entity_data
            )

            # 1. Check if all source entities exist
            source_entities_data = {}
            for entity_name in source_entities:
                node_data = await self.chunk_entity_relation_graph.get_node(entity_name)
                if not node_data:
                    print(f"Source entity '{entity_name}' does not exist")
                    continue
                    # raise ValueError(f"Source entity '{entity_name}' does not exist")
                source_entities_data[entity_name] = node_data

            # 2. Check if target entity exists and get its data if it does
            target_exists = await self.chunk_entity_relation_graph.has_node(
                target_entity
            )
            target_entity_data = {}
            if target_exists:
                target_entity_data = await self.chunk_entity_relation_graph.get_node(
                    target_entity
                )
                logger.info(
                    f"Target entity '{target_entity}' already exists, will merge data"
                )

            # 3. Merge entity data
            merged_entity_data = self._merge_entity_attributes(
                list(source_entities_data.values())
                + ([target_entity_data] if target_exists else []),
                merge_strategy,
            )

            # Apply any explicitly provided target entity data (overrides merged data)
            for key, value in target_entity_data.items():
                merged_entity_data[key] = value

            # 4. Get all relationships of the source entities
            all_relations = []
            for entity_name in source_entities:
                # Get all relationships where this entity is the source
                outgoing_edges = await self.chunk_entity_relation_graph.get_node_edges(
                    entity_name
                )
                if outgoing_edges:
                    for src, tgt in outgoing_edges:
                        # Ensure src is the current entity
                        if src == entity_name:
                            edge_data = await self.chunk_entity_relation_graph.get_edge(
                                src, tgt
                            )
                            all_relations.append(("outgoing", src, tgt, edge_data))

                # Get all relationships where this entity is the target
                incoming_edges = []
                all_labels = await self.chunk_entity_relation_graph.get_all_labels()
                for label in all_labels:
                    if label == entity_name:
                        continue
                    node_edges = await self.chunk_entity_relation_graph.get_node_edges(
                        label
                    )
                    for src, tgt in node_edges or []:
                        if tgt == entity_name:
                            incoming_edges.append((src, tgt))

                for src, tgt in incoming_edges:
                    edge_data = await self.chunk_entity_relation_graph.get_edge(
                        src, tgt
                    )
                    all_relations.append(("incoming", src, tgt, edge_data))

            # 5. Create or update the target entity
            if not target_exists:
                await self.chunk_entity_relation_graph.upsert_node(
                    target_entity, merged_entity_data
                )
                logger.info(f"Created new target entity '{target_entity}'")
            else:
                await self.chunk_entity_relation_graph.upsert_node(
                    target_entity, merged_entity_data
                )
                logger.info(f"Updated existing target entity '{target_entity}'")

            # 6. Recreate all relationships, pointing to the target entity
            relation_updates = {}  # Track relationships that need to be merged

            for rel_type, src, tgt, edge_data in all_relations:
                new_src = target_entity if src in source_entities else src
                new_tgt = target_entity if tgt in source_entities else tgt

                # Skip relationships between source entities to avoid self-loops
                if new_src == new_tgt:
                    logger.info(
                        f"Skipping relationship between source entities: {src} -> {tgt} to avoid self-loop"
                    )
                    continue

                # Check if the same relationship already exists
                relation_key = f"{new_src}|{new_tgt}"
                if relation_key in relation_updates:
                    # Merge relationship data
                    existing_data = relation_updates[relation_key]["data"]
                    merged_relation = self._merge_relation_attributes(
                        [existing_data, edge_data],
                        {
                            "description": "concatenate",
                            "keywords": "join_unique",
                            "source_id": "join_unique",
                            "weight": "max",
                        },
                    )
                    relation_updates[relation_key]["data"] = merged_relation
                    logger.info(
                        f"Merged duplicate relationship: {new_src} -> {new_tgt}"
                    )
                else:
                    relation_updates[relation_key] = {
                        "src": new_src,
                        "tgt": new_tgt,
                        "data": edge_data.copy(),
                    }

            # Apply relationship updates
            for rel_data in relation_updates.values():
                await self.chunk_entity_relation_graph.upsert_edge(
                    rel_data["src"], rel_data["tgt"], rel_data["data"]
                )
                logger.info(
                    f"Created or updated relationship: {rel_data['src']} -> {rel_data['tgt']}"
                )

            # 7. Update entity vector representation
            description = merged_entity_data.get("description", "")
            source_id = merged_entity_data.get("source_id", "")
            entity_type = merged_entity_data.get("entity_type", "")
            content = target_entity + "\n" + description

            entity_id = compute_mdhash_id(target_entity, prefix="ent-")
            entity_data_for_vdb = {
                entity_id: {
                    "content": content,
                    "entity_name": target_entity,
                    "source_id": source_id,
                    "description": description,
                    "entity_type": entity_type,
                }
            }

            await self.entities_vdb.upsert(entity_data_for_vdb)

            # 8. Update relationship vector representations
            for rel_data in relation_updates.values():
                src = rel_data["src"]
                tgt = rel_data["tgt"]
                edge_data = rel_data["data"]

                description = edge_data.get("description", "")
                keywords = edge_data.get("keywords", "")
                source_id = edge_data.get("source_id", "")
                weight = float(edge_data.get("weight", 1.0))

                content = f"{keywords}\t{src}\n{tgt}\n{description}"
                relation_id = compute_mdhash_id(src + tgt, prefix="rel-")

                relation_data_for_vdb = {
                    relation_id: {
                        "content": content,
                        "src_id": src,
                        "tgt_id": tgt,
                        "source_id": source_id,
                        "description": description,
                        "keywords": keywords,
                        "weight": weight,
                    }
                }

                await self.relationships_vdb.upsert(relation_data_for_vdb)

            # 9. Delete source entities
            for entity_name in source_entities:
                # Delete entity node
                await self.chunk_entity_relation_graph.delete_node(entity_name)
                # Delete record from vector database
                entity_id = compute_mdhash_id(entity_name, prefix="ent-")
                await self.entities_vdb.delete([entity_id])
                logger.info(f"Deleted source entity '{entity_name}'")

            # 10. Save changes
            await self._merge_entities_done()

            logger.info(
                f"Successfully merged {len(source_entities)} entities into '{target_entity}'"
            )
            return await self.get_entity_info(target_entity, include_vector_data=True)

        except Exception as e:
            logger.error(f"Error merging entities: {e}")
            raise

    def merge_entities(
        self,
        source_entities: list[str],
        target_entity: str,
        merge_strategy: dict[str, str] = None,
        target_entity_data: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """Synchronously merge multiple entities into one entity.

        Merges multiple source entities into a target entity, handling all relationships,
        and updating both the knowledge graph and vector database.

        Args:
            source_entities: List of source entity names to merge
            target_entity: Name of the target entity after merging
            merge_strategy: Merge strategy configuration, e.g. {"description": "concatenate", "entity_type": "keep_first"}
            target_entity_data: Dictionary of specific values to set for the target entity,
                overriding any merged values, e.g. {"description": "custom description", "entity_type": "PERSON"}

        Returns:
            Dictionary containing the merged entity information
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.amerge_entities(
                source_entities, target_entity, merge_strategy, target_entity_data
            )
        )

    def _merge_entity_attributes(
        self, entity_data_list: list[dict[str, Any]], merge_strategy: dict[str, str]
    ) -> dict[str, Any]:
        """Merge attributes from multiple entities.

        Args:
            entity_data_list: List of dictionaries containing entity data
            merge_strategy: Merge strategy for each field

        Returns:
            Dictionary containing merged entity data
        """
        merged_data = {}

        # Collect all possible keys
        all_keys = set()
        for data in entity_data_list:
            all_keys.update(data.keys())

        # Merge values for each key
        for key in all_keys:
            # Get all values for this key
            values = [data.get(key) for data in entity_data_list if data.get(key)]

            if not values:
                continue

            # Merge values according to strategy
            strategy = merge_strategy.get(key, "keep_first")

            if strategy == "concatenate":
                merged_data[key] = "\n\n".join(values)
            elif strategy == "keep_first":
                merged_data[key] = values[0]
            elif strategy == "keep_last":
                merged_data[key] = values[-1]
            elif strategy == "join_unique":
                # Handle fields separated by GRAPH_FIELD_SEP
                unique_items = set()
                for value in values:
                    items = value.split(GRAPH_FIELD_SEP)
                    unique_items.update(items)
                merged_data[key] = GRAPH_FIELD_SEP.join(unique_items)
            else:
                # Default strategy
                merged_data[key] = values[0]

        return merged_data

    def _merge_relation_attributes(
        self, relation_data_list: list[dict[str, Any]], merge_strategy: dict[str, str]
    ) -> dict[str, Any]:
        """Merge attributes from multiple relationships.

        Args:
            relation_data_list: List of dictionaries containing relationship data
            merge_strategy: Merge strategy for each field

        Returns:
            Dictionary containing merged relationship data
        """
        merged_data = {}

        # Collect all possible keys
        all_keys = set()
        for data in relation_data_list:
            all_keys.update(data.keys())

        # Merge values for each key
        for key in all_keys:
            # Get all values for this key
            values = [
                data.get(key)
                for data in relation_data_list
                if data.get(key) is not None
            ]

            if not values:
                continue

            # Merge values according to strategy
            strategy = merge_strategy.get(key, "keep_first")

            if strategy == "concatenate":
                merged_data[key] = "\n\n".join(str(v) for v in values)
            elif strategy == "keep_first":
                merged_data[key] = values[0]
            elif strategy == "keep_last":
                merged_data[key] = values[-1]
            elif strategy == "join_unique":
                # Handle fields separated by GRAPH_FIELD_SEP
                unique_items = set()
                for value in values:
                    items = str(value).split(GRAPH_FIELD_SEP)
                    unique_items.update(items)
                merged_data[key] = GRAPH_FIELD_SEP.join(unique_items)
            elif strategy == "max":
                # For numeric fields like weight
                try:
                    merged_data[key] = max(float(v) for v in values)
                except (ValueError, TypeError):
                    merged_data[key] = values[0]
            else:
                # Default strategy
                merged_data[key] = values[0]

        return merged_data

    async def _merge_entities_done(self) -> None:
        """Callback after entity merging is complete, ensures updates are persisted"""
        await asyncio.gather(
            *[
                cast(StorageNameSpace, storage_inst).index_done_callback()
                for storage_inst in [  # type: ignore
                    self.entities_vdb,
                    self.relationships_vdb,
                    self.chunk_entity_relation_graph,
                ]
            ]
        )

    async def get_entity_info(
        self, entity_name: str, include_vector_data: bool = False
    ) -> Dict[str, Union[str, None, Dict[str, str]]]:
        """Get detailed information of an entity

        Args:
            entity_name: Entity name (no need for quotes)
            include_vector_data: Whether to include data from the vector database

        Returns:
            dict: A dictionary containing entity information, including:
                - entity_name: Entity name
                - source_id: Source document ID
                - graph_data: Complete node data from the graph database
                - vector_data: (optional) Data from the vector database
        """

        # Get information from the graph
        node_data = await self.chunk_entity_relation_graph.get_node(entity_name)
        source_id = node_data.get("source_id") if node_data else None

        result: Dict[str, Union[str, None, Dict[str, str]]] = {
            "entity_name": entity_name,
            "source_id": source_id,
            "graph_data": node_data,
        }

        # Optional: Get vector database information
        if include_vector_data:
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            vector_data = self.entities_vdb._client.get([entity_id])
            result["vector_data"] = vector_data[0] if vector_data else None

        return result