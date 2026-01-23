"""
Evaluation utilities for ArtRAG inference results

Contains functions for evaluating generated descriptions against ground truth,
including text cleaning, batch evaluation, and metric calculation.
"""

import os
import ast
import json
import re
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from pprint import pprint

import language_evaluation
import clip
from . import clip_score

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def clean_description(description: str) -> str:
    """
    Post-process generated text by removing irrelevant symbols and formatting.

    Args:
        description: Raw generated description text

    Returns:
        str: Cleaned description text
    """
    # Remove markdown headers and formatting
    description = re.sub(r'###\s*Description of.*\n', '', description)
    description = re.sub(r'\*\*Content\*\*:', '', description)
    description = re.sub(r'\*\*Context\*\*:', '', description)
    description = re.sub(r'\*\*Form\*\*:', '', description)

    # Remove newline characters
    description = re.sub(r'\n', ' ', description)

    # Remove double quotes
    description = re.sub(r'"', '', description)

    # Remove other unwanted characters (e.g., **)
    description = re.sub(r'\*\*', '', description)

    # Normalize whitespace
    description = re.sub(r'\s+', ' ', description)

    # Remove leading and trailing whitespace
    description = description.strip()
    return description


def evaluate_batch(batch_predicts: List[str], batch_answers: List[List[str]], skip_spice: bool = False) -> Dict[str, float]:
    """
    Evaluate a batch of predictions against ground truth answers.

    Args:
        batch_predicts: List of predicted descriptions
        batch_answers: List of ground truth answer lists (each can contain multiple references)
        skip_spice: If True, skip SPICE metric (useful if SPICE causes memory issues)

    Returns:
        dict: Dictionary of metric scores, or None if all evaluation fails
    """
    # Try full CocoEvaluator first (includes all metrics)
    if not skip_spice:
        try:
            evaluator = language_evaluation.CocoEvaluator()
            batch_result = evaluator.run_evaluation(batch_predicts, batch_answers)
            return batch_result
        except Exception as e:
            print(f"Warning: CocoEvaluator failed (likely SPICE memory issue): {e}")
            print("Falling back to individual metrics (skipping SPICE)...")
            skip_spice = True
    
    # Fallback: compute metrics individually, skipping SPICE
    if skip_spice:
        batch_result = {}
        
        # Try BLEU, METEOR, ROUGE, CIDEr individually
        try:
            from pycocoevalcap.bleu.bleu import Bleu
            from pycocoevalcap.meteor.meteor import Meteor
            from pycocoevalcap.rouge.rouge import Rouge
            from pycocoevalcap.cider.cider import Cider
            from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
            
            tokenizer = PTBTokenizer()
            
            # Format data for evaluation
            refs = {idx: [{'caption': r} for r in ref_list] for idx, ref_list in enumerate(batch_answers)}
            cands = {idx: [{'caption': c}] for idx, c in enumerate(batch_predicts)}
            refs = tokenizer.tokenize(refs)
            cands = tokenizer.tokenize(cands)
            
            # BLEU scores
            try:
                bleu_scorer = Bleu(4)
                bleu_scores, _ = bleu_scorer.compute_score(refs, cands)
                if isinstance(bleu_scores, list):
                    batch_result['Bleu_1'] = bleu_scores[0]
                    batch_result['Bleu_2'] = bleu_scores[1]
                    batch_result['Bleu_3'] = bleu_scores[2]
                    batch_result['Bleu_4'] = bleu_scores[3]
                else:
                    batch_result['Bleu_4'] = bleu_scores
            except Exception as e:
                print(f"Warning: BLEU evaluation failed: {e}")
            
            # METEOR
            try:
                meteor_scorer = Meteor()
                meteor_score, _ = meteor_scorer.compute_score(refs, cands)
                batch_result['METEOR'] = meteor_score
            except Exception as e:
                print(f"Warning: METEOR evaluation failed: {e}")
            
            # ROUGE_L
            try:
                rouge_scorer = Rouge()
                rouge_score, _ = rouge_scorer.compute_score(refs, cands)
                batch_result['ROUGE_L'] = rouge_score
            except Exception as e:
                print(f"Warning: ROUGE evaluation failed: {e}")
            
            # CIDEr
            try:
                cider_scorer = Cider()
                cider_score, _ = cider_scorer.compute_score(refs, cands)
                batch_result['CIDEr'] = cider_score
            except Exception as e:
                print(f"Warning: CIDEr evaluation failed: {e}")
            
            # SPICE is intentionally skipped
            
            if batch_result:
                return batch_result
            else:
                print("Error: All individual metrics failed")
                return None
                
        except Exception as e:
            print(f"Error evaluating batch with fallback method: {e}")
            return None
    
    return None


def load_ground_truth_semart(data_type: str) -> tuple[Dict[str, List[str]], str]:
    """
    Load ground truth descriptions from SemArt dataset.

    Args:
        data_type: Type of SemArt dataset ("SemArtv1-content", "SemArtv1-context", "SemArtv2")

    Returns:
        tuple: (ground_truth_descriptions dict, image_dir path)
    """
    semartv1_types = {"SemArtv1-content", "SemArtv1-context"}
    if data_type in semartv1_types:
        ground_truth_file = "../../data/SemArt/semartv1_test_overlap_with_captions.csv"
        image_dir = "../../data/SemArt/Images"
    elif data_type == "SemArtv2":
        ground_truth_file = "../../data/SemArt/semartv2_test_overlap_with_captions.csv"
        image_dir = "../../data/SemArt/Images"
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    # Load ground truth descriptions
    ground_truth_descriptions = {}
    df = pd.read_csv(ground_truth_file, encoding='latin1', delimiter=';')

    for index, row in df.iterrows():
        content = ast.literal_eval(row['content'])
        form = ast.literal_eval(row['form']) if 'form' in row else []
        context = ast.literal_eval(row['context'])

        if data_type == "SemArtv1-content":
            full_description = content
        elif data_type == "SemArtv1-context":
            full_description = context
        elif data_type == "SemArtv2":
            full_description = content + form + context
        
        ground_truth_descriptions[row['IMAGE_FILE']] = full_description

    return ground_truth_descriptions, image_dir


def calculate_clip_scores(generated_descriptions: List[Dict], image_dir: str) -> Dict[str, Dict[str, float]]:
    """
    Calculate CLIP scores for generated descriptions.

    Args:
        generated_descriptions: List of generated description dictionaries
        image_dir: Directory containing images

    Returns:
        dict: Dictionary mapping image IDs to CLIP scores
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device, jit=False)

    model.eval()
    image_ids = [i["Image"] for i in generated_descriptions]
    image_paths = [os.path.join(image_dir, path) for path in image_ids]
    
    image_feats = clip_score.extract_all_images(
        image_paths, model, device, batch_size=64, num_workers=8
    )

    _, per_instance_image_text, candidate_feats = clip_score.get_clip_score(
        model, image_feats, [i['Generated Description'] for i in generated_descriptions], device
    )

    scores = {
        image_id: {'CLIPScore': float(clipscore)}
        for image_id, clipscore in zip(image_ids, per_instance_image_text)
    }
    
    print('CLIPScore: {:.4f}'.format(
        np.mean([s['CLIPScore'] for s in scores.values()])
    ))
    
    return scores


def evaluate_descriptions_semart(
    generated_descriptions_file: str,
    data_type: str,
    model_name: str,
    batch_size: int = 8,
    skip_spice: bool = False
) -> None:
    """
    Evaluate generated descriptions against SemArt ground truth.

    Args:
        generated_descriptions_file: Path to JSON file with generated descriptions
        data_type: Type of SemArt dataset
        model_name: Name of the model used (for output filename)
        batch_size: Batch size for evaluation
    """
    # Load generated descriptions
    with open(generated_descriptions_file, 'r') as f:
        generated_descriptions = json.load(f)

    # Load ground truth
    ground_truth_descriptions, image_dir = load_ground_truth_semart(data_type)

    # Prepare lists for evaluation
    predicts = []
    answers = []

    for description in generated_descriptions:
        image_id = description["Image"]
        if image_id in ground_truth_descriptions and ground_truth_descriptions[image_id] != []:
            cleaned_description = clean_description(description['Generated Description'])
            predicts.append(cleaned_description)
            answers.append(ground_truth_descriptions[image_id])
        else:
            print(f"Ground truth description not found for image: {image_id}")

    # Calculate average word length
    total_words = sum(len(predict.split()) for predict in predicts)
    average_word_length = total_words / len(answers) if answers else 0
    print("Average word length of answers: {:.2f}".format(average_word_length))

    # Calculate CLIP scores
    clip_scores = calculate_clip_scores(generated_descriptions, image_dir)

    # Evaluate using language_evaluation tool in batches
    num_samples = len(predicts)
    num_batches = (num_samples + batch_size - 1) // batch_size
    all_scores = []
    mean_scores = {}
    output_dir = os.path.dirname(generated_descriptions_file)
    mean_scores_file = os.path.join(
        output_dir, 'mean_scores_{}_{}.json'.format(model_name, timestamp)
    )

    for i in tqdm(range(num_batches), desc="Evaluating batches"):
        batch_predicts = predicts[i * batch_size:(i + 1) * batch_size]
        batch_answers = answers[i * batch_size:(i + 1) * batch_size]
        batch_result = evaluate_batch(batch_predicts, batch_answers, skip_spice=skip_spice)
        
        if batch_result is not None and len(batch_result) > 0:
            all_scores.append(batch_result)
            for metric in batch_result.keys():
                if metric not in mean_scores:
                    mean_scores[metric] = []
                mean_scores[metric].append(batch_result[metric])
            pprint(batch_result)
        else:
            print(f"Warning: Batch {i+1} evaluation failed or returned empty results")

    # Calculate mean scores for each metric
    if mean_scores:
        for metric in mean_scores.keys():
            mean_scores[metric] = np.mean(mean_scores[metric])
        print("Mean coco Scores:", mean_scores)
    else:
        print("Warning: No metrics were successfully computed. This may be due to:")
        print("  1. SPICE Java memory issues (try --skip_spice flag)")
        print("  2. Evaluation data format issues")
        print("  3. Missing evaluation dependencies")

    # Save mean scores
    with open(mean_scores_file, 'w') as f:
        json.dump(mean_scores, f, indent=4)

    print(f"All Mean scores saved to '{mean_scores_file}'.")
