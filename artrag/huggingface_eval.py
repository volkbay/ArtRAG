import evaluate
import numpy as np
from typing import Any, Dict, List, Sequence, Union


def _normalize_references(references: Sequence[Union[str, Sequence[str]]]) -> List[List[str]]:
    normalized = []
    for ref in references:
        if isinstance(ref, str):
            normalized.append([ref.strip()])
        elif isinstance(ref, Sequence):
            normalized.append([str(r).strip() for r in ref])
        else:
            normalized.append([str(ref).strip()])
    return normalized


def _safe_load_metric(metric_name: str):
    try:
        return evaluate.load(metric_name)
    except Exception as exc:
        print(f"Warning: Could not load Hugging Face metric '{metric_name}': {exc}")
        return None


def evaluate_batch(
    predictions: Sequence[str],
    references: Sequence[Union[str, Sequence[str]]],
) -> Dict[str, float]:
    """Compute metric scores using Hugging Face evaluate."""
    predictions = [str(p).strip() for p in predictions]
    references = _normalize_references(references)

    results: Dict[str, float] = {}

    bleu = _safe_load_metric("bleu")
    if bleu is not None:
        try:
            bleu_result = bleu.compute(predictions=predictions, references=references)
            if isinstance(bleu_result.get("precisions"), list):
                for idx, value in enumerate(bleu_result["precisions"][:4], start=1):
                    results[f"Bleu_{idx}"] = float(value)
            elif "bleu" in bleu_result:
                results["Bleu_4"] = float(bleu_result["bleu"])
        except Exception as exc:
            print(f"Warning: BLEU evaluation failed: {exc}")

    meteor = _safe_load_metric("meteor")
    if meteor is not None:
        try:
            meteor_result = meteor.compute(predictions=predictions, references=references)
            if "meteor" in meteor_result:
                results["METEOR"] = float(meteor_result["meteor"])
        except Exception as exc:
            print(f"Warning: METEOR evaluation failed: {exc}")

    rouge = _safe_load_metric("rouge")
    if rouge is not None:
        try:
            rouge_result = rouge.compute(predictions=predictions, references=references)
            if "rouge1" in rouge_result:
                results["ROUGE_1"] = float(rouge_result["rouge1"])
            if "rouge2" in rouge_result:
                results["ROUGE_2"] = float(rouge_result["rouge2"])
            if "rougeL" in rouge_result:
                results["ROUGE_L"] = float(rouge_result["rougeL"])
            elif "rougeLsum" in rouge_result:
                results["ROUGE_L"] = float(rouge_result["rougeLsum"])
        except Exception as exc:
            print(f"Warning: ROUGE evaluation failed: {exc}")

    cider = _safe_load_metric("cider")
    if cider is not None:
        try:
            cider_result = cider.compute(predictions=predictions, references=references)
            if "score" in cider_result:
                results["CIDEr"] = float(cider_result["score"])
        except Exception as exc:
            print(f"Warning: CIDEr evaluation failed: {exc}")

    spice = _safe_load_metric("spice")
    if spice is not None:
        try:
            spice_result = spice.compute(predictions=predictions, references=references)
            if "score" in spice_result:
                results["SPICE"] = float(spice_result["score"])
        except Exception as exc:
            print(f"Warning: SPICE evaluation failed: {exc}")

    return results
