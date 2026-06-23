import asyncio
import re
import threading
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration

from .prompt_art import PROMPTS
from .runtime_config import settings
from .utils import logger

# Tunables now live in the config center (cfg/default.yaml -> artrag.settings):
#   settings.bart_model_path     summarizer checkpoint
#   settings.device              cuda/cpu
#   settings.rerank_blurb_words  short blurb shown to the VLM listwise reranker
#   settings.context_word_budget per-rank final-context word budget (top entities
#                                keep detail; low-ranked ones are compressed harder)

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
        logger.info(f"Loading {settings.bart_model_path} on {settings.device} ...")
        _tokenizer = AutoTokenizer.from_pretrained(settings.bart_model_path)
        _model = BartForConditionalGeneration.from_pretrained(
            settings.bart_model_path,
            dtype=torch.float16 if settings.device == "cuda" else torch.float32,
        ).to(settings.device)
        _model.eval()
        logger.info("BART summarizer loaded.")
 
 
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
        if settings.bart_model_path.endswith('bart-large-cnn'):
            max_length = max(20, int(max_words * 1.4))   # words -> BART output tokens
            min_length = min(10, max_length // 4)
            summary = await asyncio.to_thread(_generate_sync, text, max_length, min_length)
            logger.debug("Summarized %d -> %d words", len(text.split()), len(summary.split()))
        elif settings.bart_model_path.endswith('InternVL3-14B'):
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


async def rerank_nodes_with_vlm(
    image_path: str,
    query_text: Dict[str, str],
    nodes: List[Dict[str, str]],
    use_model_func,
):
    """
    Uses a Vision-Language Model (VLM) to perform listwise reranking of nodes.
    Returns nodes in a ranked order.
    """

    # Build the query text
    painting_metadata= extract_metadata(query_text)

    # Build listwise ranking prompt
    ranking_prompt = PROMPTS["rerank_entities"]
    ranking_prompt = ranking_prompt.format(
        Metadata=painting_metadata,
        entities=[
            f"{i+1}. {node['entity_name']}: {node.get('rerank_blurb', node['description'])}"
            for i, node in enumerate(nodes)
        ],
    )
    # Call the VLM with the listwise ranking prompt (prompt itself is static -> not logged)
    response = await use_model_func(ranking_prompt, query_image_path=image_path)
    logger.debug("VLM ranking response: %s", response)

    try:
        # Extract ranked order from the model's response
        ranked_indices = [int(num.strip()) - 1 for num in response.split(',') if num.strip().isdigit()]
        ranked_nodes = [nodes[i] for i in ranked_indices if 0 <= i < len(nodes)]
    except Exception as e:
        logger.warning(f"Failed to parse VLM ranking response: {e}")
        ranked_nodes = nodes  # If ranking fails, return the original list
    # An empty/garbage response (e.g. the VLM OOM'd and fell back to text-only with
    # no parseable order) yields no ranked nodes -> fall back to the retrieval order
    # so downstream scoring isn't handed an empty set.
    if not ranked_nodes:
        logger.warning("VLM ranking produced no order; falling back to retrieval order")
        ranked_nodes = nodes
    # Create a dictionary to store VLM rank scores (higher is better)
    vlm_rank_scores = {node["entity_name"]: len(nodes) - i for i, node in enumerate(ranked_nodes)}

    return ranked_nodes, vlm_rank_scores


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
    Performs dual-passage reranking by considering:
    1. Vision-Language Model (VLM) listwise ranking
    2. Node degree from the knowledge graph (graph importance)

    Args:
        image_path: Path to the image
        painting_metadata: Metadata about the painting
        nodes: List of nodes to rerank
        use_model_func: Function to use the model
        knowledge_graph_inst: Knowledge graph instance
        topn: Number of top results to return
        vlm_weight: Weight for VLM scores (0-1). Node degree weight will be (1-vlm_weight)

    Returns a final reranked list combining both scores with specified weights.
    """
    if not nodes:
        return []

    # Build a short, length-normalized blurb for the listwise reranker WITHOUT
    # mutating each node's full description (which still feeds the final context).
    # This keeps the ranking prompt compact and prevents a verbose entity from
    # winning the ranking just because it has more text.
    # Candidates entering rerank: name, graph degree, and retrieval similarity.
    # Full descriptions are NOT dumped here (they reach the logfile once, post-budget).
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
    for node in nodes:
        node["rerank_blurb"] = await summarize_description(
            node.get("description", "UNKNOWN"), use_model_func, max_words=settings.rerank_blurb_words
        )
        if "source_id" in node:
            del node["source_id"]
    # Step 1: Get VLM listwise ranking scores
    vlm_ranked_nodes, vlm_rank_scores = await rerank_nodes_with_vlm(image_path, painting_metadata, nodes, use_model_func )
    logger.info("VLM ranking: %s", " > ".join(n["entity_name"] for n in vlm_ranked_nodes))
    # Step 2: Compute node degree ranking scores
    degree_scores = {node["entity_name"]: node["rank"] for node in nodes}

    vlm_rank_scores = softmax_normalize(vlm_rank_scores)
    degree_scores = softmax_normalize(degree_scores)

    # Step 3: Combine rankings with specified weights
    degree_weight = 1 - vlm_weight
    final_scores = {
        node["entity_name"]: vlm_weight * vlm_rank_scores.get(node["entity_name"], 0) +
                            degree_weight * degree_scores.get(node["entity_name"], 0)
        for node in nodes
    }
    # Step 4: Sort nodes based on final combined score
    final_reranked_nodes = sorted(nodes, key=lambda x: final_scores[x["entity_name"]], reverse=True)[0:topn]
    logger.info(
        "Final ranking (top %d): %s",
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
    # Final, budgeted descriptions -> logfile only (DEBUG), once.
    for node in final_reranked_nodes:
        logger.debug("Final context [%s]: %s", node["entity_name"], node["description"])

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
