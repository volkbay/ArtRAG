#!/usr/bin/env python
"""
Main script for running ArtRAG inference and evaluation on SemArt dataset.

This script supports both traditional and agentic reasoning queries,
with support for OpenAI and Bedrock LLM models.
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=".env", override=False)

# Add project root to path
sys.path.append(os.path.abspath('.'))

from artrag.llm import gpt_4o_mini_complete, gpt_4o_complete, bedrock_complete, create_bedrock_complete
from artrag.inference_utils import run_ArtRAG_inference
from artrag.evaluation import evaluate_descriptions_semart, evaluate_descriptions_artpedia


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run LightRAG inference and evaluate generated descriptions."
    )
    
    # Model and data configuration
    parser.add_argument(
        '--working_dir', 
        type=str,
        default="./built_graph/All_gpt_4o_mini_prompt_tuning_style_event_clean",
        help='Working directory for LightRAG (contains built graph).'
    )
    parser.add_argument(
        '--llm_model_func',
        type=str, 
        default='bedrock_complete',
        choices=['gpt_4o_mini_complete', 'gpt_4o_complete', 'bedrock_complete'],
        help='LLM model function to use.'
    )
    parser.add_argument(
        '--use_agentic',
        action='store_true',
        help='Use agentic reasoning query method (requires aquery_with_agentic_reasoning support)'
    )
    parser.add_argument(
        '--planner_mode',
        type=str,
        default="full",
        choices=["full", "none", "random", "text_only"],
        help='Planner mode for agentic reasoning: '
             '"full" (default, uses VLM if available), '
             '"none" (no planning, uses defaults), '
             '"random" (random planning), '
             '"text_only" (LLM-only planning, no VLM). '
             'Only used when --use_agentic is set.'
    )
    parser.add_argument(
        '--data_type',
        type=str,
        default="SemArtv2",
        choices=["SemArtv2", "Artpedia"],
        help='Dataset type to use for inference.'
    )
    
    # Inference settings
    parser.add_argument(
        '--shot_number',
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help='Number of shots for in-context learning'
    )
    parser.add_argument(
        '--retrieval_strategy',
        type=str,
        default="local",
        choices=['local', 'global', 'hybrid', 'naive', 'no-rag'],
        help='Retrieval strategy to use.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size for evaluation. Smaller values (1-4) reduce SPICE memory errors. Default: 4'
    )
    parser.add_argument(
        '--data_num',
        type=int,
        default=100,
        help='Number of data samples to process'
    )
    
    # Input/Output settings
    parser.add_argument(
        '--generated_descriptions',
        type=str,
        help='Pre-generated caption file path (if exists, inference will be skipped)'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default=None,
        help='Directory of images, with the filenames as image ids. '
             'Defaults: SemArtv2 -> ../../data/SemArt/Images, '
             'Artpedia -> ../../data/Artpedia/Images'
    )
    
    # Processing options
    parser.add_argument(
        '--mp',
        action='store_true',
        help='Enable multiprocessing (currently not functional)'
    )
    parser.add_argument(
        '--fewshot_type',
        type=str,
        default="SM_fewshot",
        choices=['MM_fewshot', 'SM_fewshot'],
        help='Type of few-shot learning examples: MultiModal (MM) or SingleModal (SM)'
    )
    parser.add_argument(
        '--question_type',
        type=str,
        default="description",
        choices=["description", "cultural&histroical", "Theme", "style&technique",
                 "Movement&school", "artist"],
        help='Type of question to generate.'
    )
    parser.add_argument(
        '--vlm_weight',
        type=float,
        default=0.5,
        help='Weight for VLM scores'
    )
    parser.add_argument(
        '--bedrock_model',
        type=str,
        default="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        help='Bedrock model ID to use (only used when llm_model_func is bedrock_complete). '
             'Examples: "us.anthropic.claude-haiku-4-5-20251001-v1:0", '
             '"us.anthropic.claude-sonnet-4-5-20250929-v1:0", '
             '"google.gemma-3-27b-it", "mistral.mistral-large-3-675b-instruct"'
    )

    # Batch processing options
    parser.add_argument(
        '--use_batch_processing',
        action='store_true',
        help='Enable async batch processing for faster inference (processes multiple queries in parallel)'
    )
    parser.add_argument(
        '--inference_batch_size',
        type=int,
        default=10,
        help='Batch size for async inference processing (default: 10). Number of rows to process in each batch. Only used if --use_batch_processing is set.'
    )
    parser.add_argument(
        '--max_concurrent',
        type=int,
        default=5,
        help='Maximum concurrent queries for async processing (default: 5). Adjust based on API rate limits. Only used if --use_batch_processing is set.'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    print(f"Arguments: {args}")

    # Map llm_model_func argument to actual function
    # For bedrock_complete, create a configured function with the specified model
    if args.llm_model_func == 'bedrock_complete':
        llm_model_func = create_bedrock_complete(args.bedrock_model)
    else:
        llm_model_func_map = {
            'gpt_4o_mini_complete': gpt_4o_mini_complete,
                'gpt_4o_complete': gpt_4o_complete,
        }
        llm_model_func = llm_model_func_map[args.llm_model_func]

    # Run inference or use pre-generated file
    if not args.generated_descriptions:
        generated_descriptions_file = run_ArtRAG_inference(
            args.working_dir,
            llm_model_func,
            args
        )
    else:
        generated_descriptions_file = args.generated_descriptions

    # Evaluate descriptions - route to appropriate evaluation function
    print(f"Batch size: {args.batch_size}")
    if args.data_type == "Artpedia":
        evaluate_descriptions_artpedia(
            generated_descriptions_file,
            args.data_type,
            args.llm_model_func,
            args.batch_size
        )
    else:  # SemArtv2
        evaluate_descriptions_semart(
            generated_descriptions_file,
            args.data_type,
            args.llm_model_func,
            args.batch_size
        )


if __name__ == "__main__":
    main()
