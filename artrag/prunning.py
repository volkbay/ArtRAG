import asyncio
import re
import threading
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration

from .prompt_art import PROMPTS
from .runtime_config import settings
from .utils import log_gpu_memory, logger

# Tunables now live in the config center (cfg/default.yaml -> artrag.settings):
#   settings.summarizer_model_path  summarizer checkpoint
#   settings.device                 cuda/cpu
#   settings.rerank_blurb_words     short blurb shown to the VLM listwise reranker
#   settings.context_word_budget    per-rank final-context word budget (top entities
#                                   keep detail; low-ranked ones are compressed harder)
#   settings.rerank_max_attempts    VLM rerank retries on missing/duplicate indices

_model = None
_tokenizer = None
_load_lock = threading.Lock()


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return
    with _load_lock:
        if _model is not None:  # re-check after acquiring lock
            return
        logger.info(f"Loading {settings.summarizer_model_path} on {settings.device} ...")
        _tokenizer = AutoTokenizer.from_pretrained(settings.summarizer_model_path)
        _model = BartForConditionalGeneration.from_pretrained(
            settings.summarizer_model_path,
            dtype=torch.float16 if settings.device == "cuda" else torch.float32,
        ).to(settings.device)
        _model.eval()
        logger.info("Summarizer loaded.")
        log_gpu_memory("after summarizer load")
 
 
def _is_degenerate(summary: str, original: str) -> bool:
    """Catch the failure modes hit with InternVL3: digit-repetition
    loops, empty output, or a 'summary' that's just as long as the input."""
    if not summary or len(summary.strip()) == 0:
        return True
    # repeated-digit or repeated-character degenerate loop, e.g. '30303030'
    if re.fullmatch(r"(\d)\1{3,}", summary.strip()):
        return True
    if re.search(r"(\d{2,4})\1{2,}", summary.strip()):
        return True
    # collapsed to a handful of words when input was substantial
    if len(summary.split()) < 3 and len(original.split()) > 15:
        return True
    return False
 
 
def _generate_sync(text: str, max_length: int, min_length: int) -> str:
    _load_model()
    inputs = _tokenizer(
        [text],
        max_length=1024,        # BART's encoder context limit
        truncation=True,
        return_tensors="pt",
    ).to(settings.device)
 
    with torch.no_grad():
        summary_ids = _model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=4,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
            forced_bos_token_id=0,
        )
 
    return _tokenizer.batch_decode(
        summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0].strip()


async def summarize_description(text, use_model_func=None, max_words=70):
    """
    Summarizes a long description down to roughly `max_words` words.
    Uses use_model_func for compatibility with the existing async pipeline.
    `max_words` lets callers request rank-aware budgets: top entities keep more
    detail, low-ranked entities are compressed harder.
    """
    global  _model
    if len(text.split()) <= max_words:  # If already within budget, return as is
        logger.debug("Summarize skipped (already <= %d words)", max_words)
        return text

    try:
        if settings.summarizer_model_path.endswith('bart-large-cnn'):
            max_length = max(20, int(max_words * 1.4))   # words -> BART output tokens
            # Scale the floor with the budget instead of hard-capping at 10 tokens
            # (the old `min(10, ...)` made BART stop far too early -> very short
            # summaries). Keep it strictly below max_length.
            min_length = min(max_length - 1, max(16, max_length // 2))
            summary = await asyncio.to_thread(_generate_sync, text, max_length, min_length)
            logger.debug("Summarized %d -> %d words", len(text.split()), len(summary.split()))
        elif settings.summarizer_model_path.endswith('InternVL3-14B'):
            # prompt = f"Summarize the following text within 30 words, keeping only the most relevant information:\n\n{text}"
            prompt = f"Summarize the following text in short sentences, keeping only the most relevant information:\n\n{text}"
            summary = await use_model_func(prompt)
            summary = summary.strip()

        if _is_degenerate(summary, text):
            logger.warning(
                f"BART produced a degenerate summary for input of "
                f"{len(text.split())} words: {summary!r}. Falling back to truncated original."
            )
            summary = " ".join(text.split())
 
    except Exception as e:
        logger.warning(f"Summarization failed: {e}")
        summary = text  # fall back to the original text if summarization fails
    
    return summary


# Normalize scores to 0-1 scale for fair combination
def min_max_normalize(scores):
    values = np.array(list(scores.values()), dtype=np.float32)
    if values.max() - values.min() > 0:
        values = (values - values.min()) / (values.max() - values.min())  # Min-max normalization
    else:
        values[:] = 1  # If all values are identical, set to 1
    return {key: value for key, value in zip(scores.keys(), values)}

def rank_normalize(scores_dict):
    """Map values to [0, 1] by their RANK (ties share the average rank), not magnitude.

    Lowest -> 0, highest -> 1. Unlike min-max, this is robust to skewed / heavy-tailed
    distributions: a single very high-degree hub no longer saturates the scale and crushes
    every other node toward 0. It also turns a count signal (node degree) into a uniform
    ordinal one, directly comparable to the VLM rank (which is already an evenly-spaced
    ordinal ramp) — so the weighted blend mixes two like-for-like distributions.
    """
    if not scores_dict:
        return {}
    keys = list(scores_dict.keys())
    vals = np.asarray([scores_dict[k] for k in keys], dtype=np.float64)
    n = len(vals)
    if n == 1:
        return {keys[0]: 1.0}
    order = np.argsort(vals, kind="mergesort")
    sorted_vals = vals[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:  # assign average rank to ties
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return {k: float(r / (n - 1)) for k, r in zip(keys, ranks)}


def softmax_normalize(scores_dict):
    """
    Applies softmax normalization to a dictionary of scores.
    Ensures smoother differentiation, making top scores more prominent.
    """
    if not scores_dict:
        return {}
    scores = np.array(list(scores_dict.values()), dtype=np.float32)

    # Avoiding overflow issues by subtracting max before exponentiation
    exp_scores = np.exp(scores - np.max(scores))
    softmax_scores = exp_scores / np.sum(exp_scores)

    return {key: score for key, score in zip(scores_dict.keys(), softmax_scores)}

def extract_metadata(text):
    match = re.search(r"Metadata:\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text  # This is already metadata without prompt


def _parse_ranking(response: str, n: int):
    """Parse a VLM ranking response into a clean 0-based index order.

    Returns ``(order, n_duplicates, n_out_of_range)`` where ``order`` keeps only
    the first occurrence of each valid 0-based index (1..n in the response). The
    two counts let the caller decide whether to retry.
    """
    order: List[int] = []
    seen = set()
    n_duplicates = 0
    n_out_of_range = 0
    for tok in re.findall(r"-?\d+", response or ""):
        idx = int(tok) - 1  # response is 1-based
        if idx < 0 or idx >= n:
            n_out_of_range += 1
            continue
        if idx in seen:
            n_duplicates += 1
            continue
        seen.add(idx)
        order.append(idx)
    return order, n_duplicates, n_out_of_range


async def rerank_nodes_with_vlm(
    image_path: str,
    query_text: Dict[str, str],
    nodes: List[Dict[str, str]],
    use_model_func,
):
    """
    Uses a Vision-Language Model (VLM) to perform listwise reranking of nodes.

    The VLM is asked for a full permutation of 1..N, but in practice (especially
    as N grows past a handful) it drops indices or repeats them. We therefore:
      * parse defensively, dropping duplicates and out-of-range numbers,
      * retry up to ``settings.rerank_max_attempts`` while the order is incomplete,
      * and finally *repair* any still-missing indices by appending them in
        retrieval order, so the returned ranking is ALWAYS a complete, duplicate-
        free permutation of all nodes (every node gets a VLM score).
    Returns ``(ranked_nodes, vlm_rank_scores)``.
    """
    n = len(nodes)
    painting_metadata = extract_metadata(query_text)
    ranking_prompt = PROMPTS["rerank_entities"].format(
        n_entities=n,
        Metadata=painting_metadata,
        entities=[
            f"{i+1}. {node['entity_name']}: {node.get('rerank_blurb', node['description'])}"
            for i, node in enumerate(nodes)
        ],
    )

    max_attempts = max(1, int(getattr(settings, "rerank_max_attempts", 3)))
    order: List[int] = []
    last_dups = last_oor = 0
    attempts = 0
    for attempts in range(1, max_attempts + 1):
        try:
            # The prompt is static -> not logged; only the response is.
            response = await use_model_func(ranking_prompt, query_image_path=image_path)
        except Exception as e:
            logger.warning("VLM rerank call failed (attempt %d/%d): %s", attempts, max_attempts, e)
            response = ""
        logger.debug("VLM ranking response (attempt %d): %s", attempts, response)
        order, last_dups, last_oor = _parse_ranking(response, n)
        missing = n - len(order)
        if missing == 0 and last_dups == 0 and last_oor == 0:
            if attempts > 1:
                logger.info("VLM ranking valid after %d attempts", attempts)
            break
        logger.warning(
            "VLM ranking attempt %d/%d invalid: %d missing, %d duplicate, %d out-of-range",
            attempts, max_attempts, missing, last_dups, last_oor,
        )

    # Repair: append any indices the VLM never produced, in retrieval order, so the
    # ranking is complete and duplicate-free regardless of how the VLM behaved.
    if len(order) < n:
        present = set(order)
        missing_idx = [i for i in range(n) if i not in present]
        logger.warning(
            "VLM ranking still missing %d/%d entities after %d attempt(s); "
            "appending them in retrieval order",
            len(missing_idx), n, attempts,
        )
        order.extend(missing_idx)

    ranked_nodes = [nodes[i] for i in order]
    # Higher is better; top of the (now complete) order gets the largest value.
    vlm_rank_scores = {node["entity_name"]: n - i for i, node in enumerate(ranked_nodes)}
    return ranked_nodes, vlm_rank_scores


_TIER = {"related": 2, "neutral": 1, "unrelated": 0}
_TIER_NAME = {2: "related", 1: "neutral", 0: "unrelated"}


def _parse_labels(response: str, n: int) -> Dict[int, int]:
    """Parse "<number>: related|neutral|unrelated" lines into {0-based index -> tier}.
    Keeps the first label seen per index; ignores anything out of range."""
    labels: Dict[int, int] = {}
    for m in re.finditer(r"(\d+)\s*[:\-.\)]\s*(related|neutral|unrelated)", response or "", re.IGNORECASE):
        idx = int(m.group(1)) - 1
        if 0 <= idx < n and idx not in labels:
            labels[idx] = _TIER[m.group(2).lower()]
    return labels


async def classify_nodes_with_vlm(image_path, query_text, nodes, use_model_func):
    """Pointwise graded-relevance reranking: ask the VLM to label every entity
    related/neutral/unrelated in small batches. Batching keeps each multimodal forward
    short (a handful of blurbs + the image), so attention memory stays bounded — this
    is what lets a 60-candidate rerank run without OOM. Per-item labels are also far more
    robust to parse than a full N-permutation (no missing/duplicate-index failures).
    Returns {entity_name -> tier int}; unlabeled entities default to neutral."""
    painting_metadata = extract_metadata(query_text)
    batch_size = max(1, int(getattr(settings, "rerank_batch_size", 12)))
    tiers: Dict[str, int] = {}
    n_batches = (len(nodes) + batch_size - 1) // batch_size
    for b, start in enumerate(range(0, len(nodes), batch_size), start=1):
        batch = nodes[start : start + batch_size]
        prompt = PROMPTS["rerank_classify"].format(
            n_entities=len(batch),
            Metadata=painting_metadata,
            entities=[
                f"{i+1}. {nd['entity_name']}: {nd.get('rerank_blurb', nd['description'])}"
                for i, nd in enumerate(batch)
            ],
        )
        try:
            response = await use_model_func(prompt, query_image_path=image_path)
        except Exception as e:
            logger.warning("VLM classify call failed (batch %d/%d): %s", b, n_batches, e)
            response = ""
        labels = _parse_labels(response, len(batch))
        for i, nd in enumerate(batch):
            tiers[nd["entity_name"]] = labels.get(i, 1)
        # Readable grouping: entity name + assigned tier + the similarity already computed
        # (instead of the raw "1: unrelated" index lines the VLM returns).
        logger.debug(
            "Classify batch %d/%d:\n%s",
            b, n_batches,
            "\n".join(
                "  {name}: {tier}{sim}".format(
                    name=nd["entity_name"],
                    tier=_TIER_NAME[labels.get(i, 1)],
                    sim=f" (sim={nd['distance']:.3f})" if nd.get("distance") is not None else "",
                )
                for i, nd in enumerate(batch)
            ),
        )
        if len(labels) < len(batch):
            logger.warning(
                "Classify batch %d/%d: %d/%d entities unlabeled -> default neutral; raw response: %s",
                b, n_batches, len(batch) - len(labels), len(batch), response,
            )
    return tiers


async def _rank_full(image_path, painting_metadata, nodes, use_model_func):
    """Listwise strategy: VLM orders the whole candidate set into one ranked list."""
    vlm_ranked_nodes, vlm_rank_scores = await rerank_nodes_with_vlm(
        image_path, painting_metadata, nodes, use_model_func
    )
    logger.info("VLM ranking: %s", " > ".join(n["entity_name"] for n in vlm_ranked_nodes))
    return vlm_rank_scores, list(nodes)


async def _rank_categorical(image_path, painting_metadata, nodes, use_model_func):
    """Categorical strategy: graded-relevance tiers, fine-ordered by retrieval similarity.
    The (tier, distance) order becomes the VLM rank score (so it merges with node degree
    exactly like the listwise score). The candidate pool is ALL nodes — selection happens
    downstream by thresholding the combined final_score (so tier AND degree both decide
    what is kept), not by a hard related/neutral gate."""
    tiers = await classify_nodes_with_vlm(image_path, painting_metadata, nodes, use_model_func)
    counts = {2: 0, 1: 0, 0: 0}
    for t in tiers.values():
        counts[t] += 1
    logger.info(
        "Categorical tiers: %d related, %d neutral, %d unrelated",
        counts[2], counts[1], counts[0],
    )

    def _dist(n):
        d = n.get("distance")
        return d if d is not None else float("-inf")

    # Fine ranking: tier first, retrieval similarity as the within-tier tiebreaker.
    ordered = sorted(
        nodes, key=lambda n: (tiers.get(n["entity_name"], 1), _dist(n)), reverse=True
    )
    total = len(ordered)
    vlm_rank_scores = {n["entity_name"]: total - i for i, n in enumerate(ordered)}
    return vlm_rank_scores, list(nodes)


async def dual_passage_rerank(
    image_path: str,
    painting_metadata: Dict[str, str],
    nodes: List[Dict[str, str]],
    use_model_func,
    knowledge_graph_inst,
    vlm_weight=0.5,
    topn=5,
):
    """
    Rerank retrieved entities by combining a VLM relevance signal with knowledge-graph
    node degree. Two VLM strategies (``settings.rerank_strategy``):
      * "full"        — listwise: the VLM orders ALL candidates; top ``topn`` are kept.
      * "categorical" — pointwise graded: the VLM labels each entity related/neutral/
                        unrelated in small batches (memory-safe), fine-ordered by retrieval
                        similarity; ALL "related" entities are kept (topn is ignored).
    Both feed the same final score: min-max(vlm_rank) * vlm_weight + min-max(degree) *
    (1-vlm_weight). Returns ``(final_reranked_nodes, interconnected_edges)``.
    """
    if not nodes:
        return [], []

    # Release any reserved-but-free CUDA blocks before the heavy multimodal forward.
    # Negligible for a single OOM, but it curbs fragmentation creep across samples in
    # batched dataset generation (models are never unloaded).
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_gpu_memory(f"rerank start ({getattr(settings, 'rerank_strategy', 'full')}, {len(nodes)} candidates)")

    # Candidates entering rerank: name, graph degree, and retrieval similarity.
    # Full descriptions are NOT dumped here (they reach the JSON once, post-budget).
    logger.info(
        "Rerank candidates (%d): %s",
        len(nodes),
        ", ".join(
            f"{n['entity_name']}(deg={n.get('rank')}, sim={n.get('distance'):.3f})"
            if n.get("distance") is not None
            else f"{n['entity_name']}(deg={n.get('rank')})"
            for n in nodes
        ),
    )
    # Build a short, length-normalized blurb for the reranker WITHOUT mutating each
    # node's full description (which still feeds the final context). This keeps the
    # prompt compact and prevents a verbose entity from winning just by volume.
    for node in nodes:
        node["rerank_blurb"] = await summarize_description(
            node.get("description", "UNKNOWN"), use_model_func, max_words=settings.rerank_blurb_words
        )
        if "source_id" in node:
            del node["source_id"]

    # Step 1: VLM relevance signal -> (rank scores over all nodes, candidate pool).
    strategy = getattr(settings, "rerank_strategy", "full")
    if strategy == "categorical":
        vlm_rank_scores, candidate_pool = await _rank_categorical(
            image_path, painting_metadata, nodes, use_model_func
        )
    else:
        vlm_rank_scores, candidate_pool = await _rank_full(
            image_path, painting_metadata, nodes, use_model_func
        )

    # Step 2: Node degree (graph importance).
    degree_scores = {node["entity_name"]: node["rank"] for node in nodes}

    # Normalize both to [0, 1] by RANK (not softmax, not raw min-max). The two signals
    # have very different shapes: the VLM score is an ordinal rank (an evenly spaced ramp
    # 1..N), while node degree is a heavy-tailed count (a few hub entities with huge degree,
    # most low). Softmax is scale-sensitive/exponential and would crush the VLM ramp into a
    # 2-3 entity spike; plain min-max on the skewed degree lets one hub saturate to 1.0 and
    # compresses everyone else to ~0 (degree collapses to "is this the hub?"). Rank-
    # normalization maps each signal to a uniform [0, 1] by position, so a strong-but-not-
    # outlier node keeps real weight and vlm_weight blends two like-for-like distributions.
    vlm_rank_scores = rank_normalize(vlm_rank_scores)
    degree_scores = rank_normalize(degree_scores)

    # Step 3: Combine with the configured weights.
    degree_weight = 1 - vlm_weight
    final_scores = {
        node["entity_name"]: vlm_weight * vlm_rank_scores.get(node["entity_name"], 0) +
                            degree_weight * degree_scores.get(node["entity_name"], 0)
        for node in nodes
    }
    # Step 4: Sort the candidate pool by final score, then select.
    #   * "full"        -> keep the top `topn`.
    #   * "categorical" -> keep every node whose final_score >= rerank.score_threshold
    #                      (NOT capped by topn). Both tier and degree feed final_score,
    #                      so a strong-but-low-similarity node can still clear the bar.
    ranked = sorted(
        candidate_pool, key=lambda x: final_scores[x["entity_name"]], reverse=True
    )
    if strategy == "categorical":
        thr = float(getattr(settings, "rerank_score_threshold", 0.5))
        final_reranked_nodes = [n for n in ranked if final_scores[n["entity_name"]] >= thr]
        if not final_reranked_nodes:
            # Never return empty -> keep the single best so the context isn't blank.
            final_reranked_nodes = ranked[:1]
            logger.warning(
                "No entity cleared final_score >= %.2f; keeping best (%.3f)",
                thr, final_scores[final_reranked_nodes[0]["entity_name"]] if final_reranked_nodes else 0,
            )
        logger.info(
            "Final ranking (%d kept, categorical, thr=%.2f): %s",
            len(final_reranked_nodes), thr,
            ", ".join(f"{n['entity_name']}({final_scores[n['entity_name']]:.3f})" for n in final_reranked_nodes),
        )
    else:
        final_reranked_nodes = ranked[:topn]
        logger.info(
            "Final ranking (%d kept, full): %s",
            len(final_reranked_nodes),
            ", ".join(f"{n['entity_name']}({final_scores[n['entity_name']]:.3f})" for n in final_reranked_nodes),
        )

    # Step 5: Apply a rank-aware word budget to the descriptions that actually
    # reach the final context. Top entities keep their full detail; lower-ranked
    # (often more generic) entities are compressed harder so they cannot dominate
    # the response by volume. Short descriptions pass through untouched.
    budgets = settings.context_word_budget
    for i, node in enumerate(final_reranked_nodes):
        # Carry the (normalized) component + combined scores so downstream output
        # (the per-sample JSON) can report why each entity ranked where it did.
        name = node["entity_name"]
        node["vlm_score"] = vlm_rank_scores.get(name)
        node["degree_score"] = degree_scores.get(name)
        node["final_score"] = final_scores.get(name)
        budget = budgets[i] if i < len(budgets) else budgets[-1]
        node["description"] = await summarize_description(
            node.get("description", "UNKNOWN"), use_model_func, max_words=budget
        )
        node.pop("rerank_blurb", None)
    # Final budgeted descriptions are NOT logged here — they are written once to the
    # per-sample JSON as reranked_entities[*].description.

    inter_edges = await find_interconnected_edges([node["entity_name"] for node in final_reranked_nodes], knowledge_graph_inst)

    return final_reranked_nodes, inter_edges




async def find_interconnected_edges(all_node_names, knowledge_graph_inst):
    """
    Fetch all edges between the combined set of nodes.
    """
    all_edges = set()
    # check if there is are """ in node_name, if not add it to node name, like 'HIGH RENAISSANCE' to '"HIGH RENAISSANCE"'
    all_node_names = [f'"{node_name}"' if '"' not in node_name else node_name for node_name in all_node_names]    

    for node_name in all_node_names:
        # Check if there is are """ in node_name, if not add it to node name, like 'HIGH RENAISSANCE' to '"HIGH RENAISSANCE"'
        # if '"' not in node_name:
        #     node_name = f'"{node_name}"'
        node_edges = await knowledge_graph_inst.get_node_edges(node_name)
        all_edges.update([tuple(sorted(e)) for e in node_edges])
    all_edges = list(all_edges)
    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    all_edges_data = [
        {"src_tgt": k, **v}
        for k, v in zip(all_edges, all_edges_pack)
        if v is not None and k[0] in all_node_names and k[1] in all_node_names
    ]
    return all_edges_data
