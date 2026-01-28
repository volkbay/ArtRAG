"""
Batch processing utilities for ArtRAG inference.

This module provides async batch processing capabilities to speed up inference
by processing multiple queries in parallel while respecting API rate limits.
"""

import os
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import pandas as pd
import json
import logging

from .lightrag import LightRAG, QueryParam
from .inference_utils import build_query_text, load_dataset, extract_painting_info

logger = logging.getLogger(__name__)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
current_date = datetime.now().strftime("%Y-%m-%d")


async def process_single_row_async(
    rag: LightRAG,
    row: pd.Series,
    args: Any,
    index: int
) -> Dict[str, Any]:
    """
    Process a single row asynchronously with error handling.
    
    Args:
        rag: LightRAG instance
        row: DataFrame row
        args: Arguments object
        index: Row index (for logging)
        
    Returns:
        Result dictionary with error field if processing failed
    """
    try:
        # Row processing logged at batch level, no need for per-row debug logs
        
        # Extract painting information using unified function
        painting_info = extract_painting_info(row, args.data_type)
        img_path = painting_info["img_path"]
        metadata = painting_info["metadata"]
        
        # Validate image path
        if not os.path.exists(img_path):
            logger.warning(f"Image not found for row {index}: {img_path}, continuing without image")
            img_path = None
        
        # Build query text
        query_text = build_query_text(row, args)
        
        # Run inference
        if args.use_agentic:
            generated_description, retrieved_context, rerank_context = await run_agentic_query_async(
                rag, query_text, img_path, metadata, args.retrieval_strategy,
                vlm_weight=getattr(args, 'vlm_weight', 0.5),
                data_type=args.data_type,
                shot_number=args.shot_number,
                fewshot_type=args.fewshot_type,
                planner_mode=getattr(args, 'planner_mode', 'full')
            )
        else:
            generated_description, retrieved_context, rerank_context = await run_traditional_query_async(
                rag, query_text, img_path, args
            )
        
        # Build result dictionary - handle different dataset structures
        result = {
            'Title': painting_info.get('title', ''),
            'Image': painting_info.get('img_id', ''),
            'Author': painting_info.get('author', ''),
            'Generated Description': generated_description,
            'Retrieved context': retrieved_context,
            'rerank_context': rerank_context,
            'row_index': index
        }
        
        # Add dataset-specific fields
        if args.data_type == "Artpedia":
            result['Year'] = painting_info.get('year', '')
            result['Concepts'] = painting_info.get('tags', '')
        elif args.data_type == "SemArtv2":
            result['Technique'] = painting_info.get('technique', '')
            result['Timeframe'] = painting_info.get('timeframe', '')
            result['Concepts'] = painting_info.get('tags', '')
        
        return result
    except Exception as e:
        logger.error(f"Error processing row {index}: {e}", exc_info=True)
        # Return error result instead of crashing
        painting_info = extract_painting_info(row, args.data_type)
        result = {
            'Title': painting_info.get('title', ''),
            'Image': painting_info.get('img_id', ''),
            'Author': painting_info.get('author', ''),
            'Generated Description': f"ERROR: {str(e)}",
            'Retrieved context': None,
            'rerank_context': None,
            'row_index': index,
            'error': str(e)
        }
        
        # Add dataset-specific fields
        if args.data_type == "Artpedia":
            result['Year'] = painting_info.get('year', '')
            result['Concepts'] = painting_info.get('tags', '')
        elif args.data_type == "SemArtv2":
            result['Technique'] = painting_info.get('technique', '')
            result['Timeframe'] = painting_info.get('timeframe', '')
            result['Concepts'] = painting_info.get('tags', '')
        
        return result


async def run_agentic_query_async(
    rag: LightRAG,
    query_text: str,
    img_path: Optional[str],
    metadata: Dict[str, Any],
    mode: str,
    **kwargs
) -> tuple[str, None, None]:
    """Async version of run_agentic_query."""
    multimodal_content = (
        [{"type": "image", "img_path": img_path}]
        if img_path and os.path.exists(img_path)
        else []
    )
    
    query_kwargs = {
        "vlm_weight": kwargs.get('vlm_weight', 0.5),
        "data_type": kwargs.get('data_type', 'SemArtv2'),
        "shot_number": kwargs.get('shot_number', 1),
        "fewshot_type": kwargs.get('fewshot_type', 'SM_fewshot'),
        "planner_mode": kwargs.get('planner_mode', 'full'),
        **{k: v for k, v in kwargs.items() 
           if k not in ['vlm_weight', 'data_type', 'shot_number', 'fewshot_type', 'planner_mode']}
    }
    
    # Direct async call (no need to import from inference_utils)
    generated_description = await rag.aquery_with_agentic_reasoning(
        query=query_text,
        multimodal_content=multimodal_content,
        metadata=metadata,
        mode=mode,
        **query_kwargs
    )
    
    return generated_description, None, None


async def run_traditional_query_async(
    rag: LightRAG,
    query_text: str,
    img_path: Optional[str],
    args: Any
) -> tuple[str, Any, Any]:
    """Async version of run_traditional_query."""
    query = {"text": query_text, "image": img_path} if img_path else query_text
    param = QueryParam(mode=args.retrieval_strategy)
    param.data_type = args.data_type
    param.fewshot_type = args.fewshot_type
    param.shot_number = args.shot_number
    param.vlm_weight = args.vlm_weight
    
    response, beforererank_context, rerank_context = await rag.aquery(query, param)
    return response, beforererank_context, rerank_context


async def run_ArtRAG_inference_async(
    working_dir: str,
    llm_model_func: callable,
    args: Any,
    batch_size: int = 10,
    max_concurrent: int = 5
) -> str:
    """
    Run inference on SemArt dataset with async batch processing.
    
    Args:
        working_dir: Working directory for LightRAG
        llm_model_func: LLM model function
        args: Arguments object
        batch_size: Number of rows to process in each batch
        max_concurrent: Maximum concurrent queries (respects API rate limits)
        
    Returns:
        Path to output file
    """
    # Get model name and planner mode for logging
    model_name = getattr(args, 'llm_model_func', 'unknown')
    if isinstance(model_name, str):
        model_display = model_name
        # If using Bedrock, show the specific model ID
        if model_name == 'bedrock_complete':
            bedrock_model = getattr(args, 'bedrock_model', 'unknown')
            model_display = f"{model_name} ({bedrock_model})"
    else:
        model_display = getattr(model_name, '__name__', 'unknown')
        # Try to extract Bedrock model from closure if it's a Bedrock function
        if 'bedrock' in model_display.lower():
            try:
                if hasattr(model_name, '__closure__') and model_name.__closure__:
                    # Extract model from closure
                    closure_vars = [cell.cell_contents for cell in model_name.__closure__]
                    for var in closure_vars:
                        if isinstance(var, str) and ('anthropic' in var or 'claude' in var.lower() or 'gemma' in var.lower() or 'mistral' in var.lower()):
                            model_display = f"{model_display} ({var})"
                            break
            except Exception:
                pass
    
    planner_mode = getattr(args, 'planner_mode', 'full') if getattr(args, 'use_agentic', False) else 'N/A'
    query_method = "agentic" if getattr(args, 'use_agentic', False) else getattr(args, 'retrieval_strategy', 'unknown')
    
    # Log configuration clearly
    logger.info("=" * 80)
    logger.info("ART RAG INFERENCE CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Model: {model_display}")
    logger.info(f"Query Method: {query_method}")
    if getattr(args, 'use_agentic', False):
        logger.info(f"Planner Mode (Ablation): {planner_mode}")
    logger.info(f"Dataset: {args.data_type}")
    logger.info(f"Question Type: {args.question_type}")
    logger.info(f"Shot Number: {args.shot_number}")
    logger.info(f"Few-shot Type: {getattr(args, 'fewshot_type', 'N/A')}")
    logger.info(f"VLM Weight: {getattr(args, 'vlm_weight', 0.5)}")
    logger.info(f"Data Samples: {args.data_num}")
    logger.info(f"Batch Size: {batch_size}, Max Concurrent: {max_concurrent}")
    logger.info("=" * 80)
    
    # Initialize LightRAG
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func
    )
    
    # Load data using unified function
    data = load_dataset(args.data_type, args.data_num)
    total_rows = len(data)
    
    logger.info(f"Starting inference: Processing {total_rows} rows")
    
    # Create semaphore to limit concurrent queries
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(row, index):
        """Process row with semaphore to limit concurrency."""
        async with semaphore:
            return await process_single_row_async(rag, row, args, index)
    
    # Create tasks for all rows
    tasks = [
        process_with_semaphore(row, index)
        for index, row in data.iterrows()
    ]
    
    # Process in batches with progress tracking
    all_results = []
    num_batches = (len(tasks) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(tasks), batch_size), desc="Processing batches"):
        batch_tasks = tasks[i:i + batch_size]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Handle exceptions in batch
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}", exc_info=True)
                all_results.append({
                    'Generated Description': f"ERROR: {str(result)}",
                    'error': str(result)
                })
            else:
                all_results.append(result)
    
    # Sort by row_index to maintain order
    all_results.sort(key=lambda x: x.get('row_index', 0))
    
    # Count errors
    errors = [r for r in all_results if 'error' in r]
    if errors:
        logger.warning(f"Encountered {len(errors)} errors during processing")
    
    # Save results
    results_df = pd.DataFrame(all_results)
    output_dir = os.path.join(
        working_dir,
        f"output_{current_date}_{args.data_type}_{args.data_num}data"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    query_method = "agentic" if args.use_agentic else args.retrieval_strategy
    if args.use_agentic:
        planner_mode = getattr(args, 'planner_mode', 'full')
        query_method = f"agentic_{planner_mode}"
    model_suffix = getattr(args, 'llm_model_func', 'unknown').replace("_", "-")
    output_file = os.path.join(
        output_dir,
        f'generated_descriptions_{query_method}_{model_suffix}_{timestamp}.json'
    )
    
    results_df.to_json(output_file, orient='records', indent=4, force_ascii=False)
    
    # Save args
    args_dict = vars(args)
    args_file = os.path.join(output_dir, f'args_{timestamp}.json')
    with open(args_file, 'w') as f:
        json.dump(args_dict, f, indent=4)
    
    # Log completion summary
    logger.info("=" * 80)
    logger.info("INFERENCE COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Model: {model_display}")
    logger.info(f"Query Method: {query_method}")
    if getattr(args, 'use_agentic', False):
        logger.info(f"Planner Mode (Ablation): {planner_mode}")
    logger.info(f"Total Rows Processed: {total_rows}")
    logger.info(f"Successful: {total_rows - len(errors)}")
    if errors:
        logger.info(f"Errors: {len(errors)} (check 'error' field in results)")
    logger.info(f"Output File: {output_file}")
    logger.info("=" * 80)
    
    # Wait for all pending tasks (except the current one) to complete before returning
    # This prevents "Event loop is closed" errors from httpx/async clients
    # Get all pending tasks and wait for them to complete
    current_task = asyncio.current_task()
    pending_tasks = [task for task in asyncio.all_tasks() 
                     if task != current_task and not task.done()]
    if pending_tasks:
        # Wait for cleanup tasks (no need to log unless there's an issue)
        try:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        except Exception as e:
            # Only log if it's a real error, not just cleanup exceptions
            logger.warning(f"Cleanup tasks had exceptions: {e}")
        # Give a small additional delay for any final cleanup
        await asyncio.sleep(0.1)
    
    return output_file
