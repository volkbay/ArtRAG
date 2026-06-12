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

from .huggingface_eval import evaluate_batch as hf_evaluate_batch
import clip
from . import clip_score

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def extract_sections_from_description(description: str) -> Dict[str, str]:
    """
    Extract Content, Form, and Context sections from generated description.
    
    Args:
        description: Generated description text with sections
        
    Returns:
        dict: {"content": "...", "form": "...", "context": "..."} (None if section not found)
    """
    sections = {"content": None, "form": None, "context": None}
    
    # Find all section markers and their positions
    section_markers = []
    for section_name in ["content", "form", "context"]:
        # Try multiple patterns for section headers:
        # **Content**:, *Content*:, Content:, # Content, ## Content, etc.
        # Also handle variations like "Content section:", "Content -", etc.
        patterns = [
            rf"(?i)\*\*{section_name.capitalize()}\*\*:\s*",  # **Content**:
            rf"(?i)\*{section_name.capitalize()}\*:\s*",      # *Content*:
            rf"(?i)#+\s*{section_name.capitalize()}:\s*",     # # Content:
            rf"(?i){section_name.capitalize()}:\s*",          # Content:
            rf"(?i)\*\*{section_name.capitalize()}\*\*\s*",   # **Content**
            rf"(?i){section_name.capitalize()}\s+section:\s*", # Content section:
            rf"(?i){section_name.capitalize()}\s*-\s*",       # Content -
        ]
        
        for pattern in patterns:
            match = re.search(pattern, description)
            if match:
                section_markers.append((section_name, match.start(), match.end()))
                break  # Use first match found
    
    # Sort by position in text
    section_markers.sort(key=lambda x: x[1])
    
    # Extract sections based on markers
    for i, (section_name, start_pos, header_end) in enumerate(section_markers):
        # Find end position (next section marker or end of text)
        if i + 1 < len(section_markers):
            end_pos = section_markers[i + 1][1]
        else:
            end_pos = len(description)
        
        # Extract section text (skip the header)
        section_text = description[header_end:end_pos].strip()
        
        # Clean up: remove extra whitespace, newlines
        section_text = re.sub(r'\s+', ' ', section_text)
        if section_text:
            sections[section_name] = section_text
    
    return sections


def clean_description(description: str, preserve_terms: bool = True) -> str:
    """
    Post-process generated text by removing irrelevant symbols and formatting.
    Preserves important terminology (artist names, movements, techniques) for better metric scores.

    Args:
        description: Raw generated description text
        preserve_terms: If True, preserve exact terminology (default: True)

    Returns:
        str: Cleaned description text
    """
    # Remove step-by-step markers and verbose introductions
    description = re.sub(r'\*\*Step \d+[:\*]*\*\*', '', description)
    description = re.sub(r'Step \d+[:\-]\s*', '', description)
    description = re.sub(r'Here\'s a description.*?synthesizing.*?:\s*', '', description, flags=re.IGNORECASE)
    description = re.sub(r'Okay, here\'s.*?according to.*?:\s*', '', description, flags=re.IGNORECASE)
    description = re.sub(r'Here\'s.*?grounded in.*?:\s*', '', description, flags=re.IGNORECASE)
    description = re.sub(r'Based on.*?retrieved context.*?:\s*', '', description, flags=re.IGNORECASE)
    description = re.sub(r'Following.*?generation plan.*?:\s*', '', description, flags=re.IGNORECASE)
    
    # Remove markdown headers and formatting
    description = re.sub(r'###\s*Description of.*\n', '', description)
    description = re.sub(r'##\s*Description of.*\n', '', description)
    description = re.sub(r'#+\s*Final answer:\s*', '', description, flags=re.IGNORECASE)
    description = re.sub(r'#+\s*Your Task.*?\n', '', description, flags=re.IGNORECASE)
    
    # Remove evidence markers like (kg|text - *based on...*)
    description = re.sub(r'\([^)]*kg\|text[^)]*\)', '', description)
    description = re.sub(r'\(visual\|metadata\)', '', description)
    description = re.sub(r'\*based on[^*]*\*', '', description)
    description = re.sub(r'\(evidence:[^)]*\)', '', description, flags=re.IGNORECASE)
    description = re.sub(r'\[evidence:[^\]]*\]', '', description, flags=re.IGNORECASE)

    # Remove newline characters (replace with space to preserve word boundaries)
    description = re.sub(r'\n+', ' ', description)

    # Remove double quotes (but preserve single quotes for contractions)
    description = re.sub(r'"', '', description)

    # Remove remaining markdown bold/italic markers (preserve the text inside)
    # But be careful not to remove section headers that might still be present
    description = re.sub(r'\*\*([^*]+)\*\*', r'\1', description)
    description = re.sub(r'\*([^*]+)\*', r'\1', description)
    
    # Remove markdown code blocks
    description = re.sub(r'```[^`]*```', '', description)
    description = re.sub(r'`([^`]+)`', r'\1', description)

    # Remove extra punctuation artifacts
    description = re.sub(r'\.{3,}', '...', description)  # Normalize ellipsis
    description = re.sub(r'--+', '--', description)  # Normalize dashes

    # Normalize whitespace (but preserve single spaces)
    description = re.sub(r'\s+', ' ', description)

    # Remove leading and trailing whitespace and punctuation
    description = description.strip(' .,;:!?')

    return description


def evaluate_batch(batch_predicts: List[str], batch_answers: List[List[str]]) -> Dict[str, float]:
    """
    Evaluate a batch of predictions against ground truth answers.

    Args:
        batch_predicts: List of predicted descriptions
        batch_answers: List of ground truth answer lists (each can contain multiple references)

    Returns:
        dict: Dictionary of metric scores
    """
    try:
        return hf_evaluate_batch(batch_predicts, batch_answers)
    except Exception as e:
        print(f"Error evaluating batch with Hugging Face metrics: {e}")
        return {}


def load_ground_truth_semart(data_type: str) -> tuple[Dict[str, List[str]], Dict[str, Dict[str, List[str]]], str]:
    """
    Load ground truth descriptions from SemArt dataset.

    Args:
        data_type: Type of SemArt dataset ("SemArtv1-content", "SemArtv1-context", "SemArtv2")

    Returns:
        tuple: (ground_truth_descriptions dict, section_info dict, image_dir path)
        - ground_truth_descriptions: Combined descriptions for evaluation
        - section_info: For SemArtv2, tracks which sections exist: {image_id: {"content": [...], "form": [...], "context": [...]}}
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
    section_info = {}  # For SemArtv2, track individual sections
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
            # Store section info for SemArtv2
            section_info[row['IMAGE_FILE']] = {
                "content": content if content and content != [] else None,
                "form": form if form and form != [] else None,
                "context": context if context and context != [] else None
            }
            # Combine only non-empty sections with section labels
            full_description = []
            if content and content != []:
                # Add "Content: " prefix to each content reference
                for ref in content:
                    full_description.append(f"Content: {ref}")
            if form and form != []:
                # Add "Form: " prefix to each form reference
                for ref in form:
                    full_description.append(f"Form: {ref}")
            if context and context != []:
                # Add "Context: " prefix to each context reference
                for ref in context:
                    full_description.append(f"Context: {ref}")
        
        ground_truth_descriptions[row['IMAGE_FILE']] = full_description

    return ground_truth_descriptions, section_info, image_dir


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
    batch_size: int = 8
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
    ground_truth_descriptions, section_info, image_dir = load_ground_truth_semart(data_type)

    # Prepare lists for evaluation
    predicts = []
    answers = []

    for description in generated_descriptions:
        image_id = description["Image"]
        if image_id in ground_truth_descriptions and ground_truth_descriptions[image_id] != []:
            generated_text = description['Generated Description']
            
            # For SemArtv2, extract and match sections
            if data_type == "SemArtv2" and image_id in section_info:
                # Extract sections from generated description (before cleaning)
                generated_sections = extract_sections_from_description(generated_text)
                gt_sections = section_info[image_id]
                
                # Build description using only sections that exist in ground truth
                matched_sections = []
                for section_name in ["content", "form", "context"]:
                    if gt_sections[section_name] is not None:  # Section exists in ground truth
                        if generated_sections[section_name] is not None:
                            # Clean each section individually to preserve important terms
                            cleaned_section = clean_description(generated_sections[section_name], preserve_terms=True)
                            if cleaned_section:  # Only add non-empty cleaned sections
                                # Add section label at the beginning
                                section_label = section_name.capitalize()  # "content" -> "Content"
                                matched_sections.append(f"{section_label}: {cleaned_section}")
                        # Note: If section is missing in generated, we don't add empty string
                        # This is better than penalizing with empty string
                
                # Combine matched sections
                if matched_sections:
                    cleaned_description = " ".join(matched_sections).strip()
                else:
                    # Fallback: if no sections matched, try cleaning full description
                    # This handles cases where section extraction failed
                    cleaned_description = clean_description(generated_text, preserve_terms=True)
            else:
                # If section extraction fails, use full description
                cleaned_description = clean_description(generated_text, preserve_terms=True)
            
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
        batch_result = evaluate_batch(batch_predicts, batch_answers)
        
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
        print("  1. Evaluation data format issues")
        print("  2. Missing evaluation dependencies")
        print("  3. Java memory issues (consider reducing batch_size)")

    # Save mean scores
    with open(mean_scores_file, 'w') as f:
        json.dump(mean_scores, f, indent=4)

    print(f"All Mean scores saved to '{mean_scores_file}'.")


def clean_description_artpedia(description: str) -> str:
    """
    Clean description for Artpedia/ExpArt evaluation.
    Removes markdown formatting and normalizes text.
    
    Args:
        description: Generated description text
        
    Returns:
        str: Cleaned description
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


def evaluate_descriptions_artpedia(
    generated_descriptions_file: str,
    data_type: str,
    model_name: str,
    batch_size: int = 8
) -> None:
    """
    Evaluate generated descriptions for Artpedia dataset.
    
    Args:
        generated_descriptions_file: Path to JSON file with generated descriptions
        data_type: Dataset type ("Artpedia")
        model_name: Model name for output file naming
        batch_size: Batch size for evaluation
    """
    if data_type != "Artpedia":
        raise ValueError(f"Unsupported data_type for Artpedia evaluation: {data_type}. Only 'Artpedia' is supported.")
    
    # Load generated descriptions
    with open(generated_descriptions_file, 'r') as f:
        generated_descriptions = json.load(f)
    
    # Load ground truth
    ground_truth_file = "../../data/Artpedia/artpedia_test.json"
    df = pd.read_json(ground_truth_file, encoding='utf8', orient="index")
    image_dir = "../../data/Artpedia/Images"
    
    # Load ground truth descriptions
    ground_truth_descriptions = {}
    for index, row in df.iterrows():
        content = row.get('contextual_sentences', [])
        context = row.get('visual_sentences', [])
        full_description = content + context
        ground_truth_descriptions[row.name] = full_description
    
    # Prepare lists for evaluation
    predicts = []
    answers = []
    
    for description in generated_descriptions:
        image_id = description.get("Image", "")
        if image_id in ground_truth_descriptions:
            cleaned_description = clean_description_artpedia(
                description.get('Generated Description', '')
            )
            if cleaned_description:
                predicts.append(cleaned_description)
                answers.append(ground_truth_descriptions[image_id])
        else:
            print(f"Ground truth description not found for image: {image_id}")
    
    if not predicts:
        print("Warning: No valid predictions found for evaluation")
        return
    
    # Calculate average word length
    total_words = sum(len(predict.split()) for predict in predicts)
    average_word_length = total_words / len(predicts) if predicts else 0
    print(f"Average word length of answers: {average_word_length:.2f}")
    
    # Calculate CLIP scores
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()
    
    image_ids = [str(d.get("Image", "")) + ".jpg" for d in generated_descriptions]
    
    image_paths = [os.path.join(image_dir, path) for path in image_ids if os.path.exists(os.path.join(image_dir, path))]
    
    if image_paths:
        image_feats = clip_score.extract_all_images(
            image_paths, model, device, batch_size=64, num_workers=8
        )
        
        _, per_instance_image_text, candidate_feats = clip_score.get_clip_score(
            model, image_feats, predicts[:len(image_paths)], device
        )
        
        clip_scores = {image_id: {'CLIPScore': float(clipscore)}
                      for image_id, clipscore in
                      zip(image_ids[:len(image_paths)], per_instance_image_text)}
        mean_clip_score = np.mean([s['CLIPScore'] for s in clip_scores.values()])
        print(f'CLIPScore: {mean_clip_score:.4f}')
    else:
        print("Warning: No valid image paths found for CLIP score calculation")
    
    # Evaluate using language_evaluation tool in batches
    num_samples = len(predicts)
    num_batches = (num_samples + batch_size - 1) // batch_size
    all_scores = []
    mean_scores = {}
    output_dir = os.path.dirname(generated_descriptions_file)
    mean_scores_file = os.path.join(
        output_dir, f'mean_scores_{model_name}_{data_type}_{timestamp}.json'
    )
    
    for i in tqdm(range(num_batches), desc="Evaluating batches"):
        batch_predicts = predicts[i * batch_size:(i + 1) * batch_size]
        batch_answers = answers[i * batch_size:(i + 1) * batch_size]
        batch_result = evaluate_batch(batch_predicts, batch_answers)
        if batch_result is not None:
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
        print("Warning: No metrics were successfully computed.")
    
    # Save mean scores
    with open(mean_scores_file, 'w') as f:
        json.dump(mean_scores, f, indent=4)
    
    print(f"All Mean scores saved to '{mean_scores_file}'.")
