"""
Inference utilities for ArtRAG

Contains functions for running inference on datasets, building queries,
and handling both traditional and agentic reasoning queries.
"""

import os
import asyncio
from datetime import datetime
from typing import Dict, Any
from tqdm import tqdm
import pandas as pd
import json

from .lightrag import LightRAG, QueryParam

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
current_date = datetime.now().strftime("%Y-%m-%d")


def build_query_text(row: pd.Series, args: Any) -> str:
    """
    Build query text based on question type and available data fields.
    Supports SemArtv2 and Artpedia datasets.

    Args:
        row: DataFrame row containing painting data
        args: Arguments object with question_type and data_type

    Returns:
        str: Formatted query text
    """
    data_type = args.data_type
    
    # Handle Artpedia dataset
    if data_type == "Artpedia":
        # For Artpedia, use simpler query structure
        title = row.get('title', '')
        year = row.get('year', '')
        tags = row.get('tags', '')
        
        # Define the presence of each attribute (Artpedia structure)
        form_exists = False
        context_exists = True
        content_exists = True
        
        input_text = "Please generate the description on "
        if context_exists:
            input_text += " --context--,"
        if form_exists:
            input_text += " --form--,"
        if content_exists:
            input_text += " --content--,"
        input_text += f" perspective of the painting with Metadata:  {title},  Year: {year}, simple description: {tags}"
        return input_text
    
    # Handle SemArtv2 dataset (original logic)
    # Determine which fields exist
    content = row.get('content', '')
    context = row.get('context', '')
    form = row.get('form', '') if data_type == "SemArtv2" else None

    context_exists = pd.notna(context) and context != '[]'
    content_exists = pd.notna(content) and content != '[]'
    form_exists = pd.notna(form) and form != '[]' if form is not None else False

    # Build query based on question type
    if args.question_type == "description":
        input_text = "Please generate the description on "
        if context_exists:
            input_text += " --context--,"
        if form_exists:
            input_text += " --form--,"
        if content_exists:
            input_text += " --content--,"
        input_text += "perspective of the painting"
        input_text = input_text.rstrip(',')
    elif args.question_type == "cultural&histroical":
        input_text = "What historical events or cultural movements influenced the creation of this painting? "
    elif args.question_type == "Theme":
        input_text = "What themes or beliefs are reflected in this painting? "
    elif args.question_type == "style&technique":
        input_text = "Does the painting reflect a shift from one stylistic period to another? "
    elif args.question_type == "Movement&school":
        input_text = "How does this painting embody the principles of its art movement? "
    elif args.question_type == "artist":
        input_text = "How does this painting reflect the artist's personal beliefs or experiences? "
    else:
        input_text = "Please describe this painting."

    # Add metadata
    title = row.get('TITLE', '')
    author = row.get('AUTHOR', '')
    technique = row.get('TECHNIQUE', '')
    timeframe = row.get('TIMEFRAME', '')
    tags = row.get('tags', '')
    
    input_text += f"with painting Metadata:  {title}, Author: {author}, Technique: {technique}, Timeframe: {timeframe}, Painting Concepts: {tags}"
    
    return input_text


def get_data_path(data_type: str) -> str:
    """
    Get the path to dataset file based on data type.
    Supports SemArtv2 (CSV) and Artpedia (JSON).

    Args:
        data_type: Type of dataset ("SemArtv2" or "Artpedia")

    Returns:
        str: Path to dataset file
    """
    if data_type == "SemArtv2":
        return "../../data/SemArt/semartv2_test_overlap_with_captions.csv"
    elif data_type == "Artpedia":
        return "../../data/Artpedia/artpedia_test.json"
    else:
        raise ValueError(f"Unknown data_type: {data_type}. Supported: SemArtv2, Artpedia")




def load_dataset(data_type: str, data_num: int = 100):
    """
    Load dataset based on data type.
    Supports SemArtv2 (CSV) and Artpedia (JSON).

    Args:
        data_type: Type of dataset ("SemArtv2" or "Artpedia")
        data_num: Number of samples to load

    Returns:
        pd.DataFrame: Loaded dataset
    """
    data_path = get_data_path(data_type)
    
    if data_type == "Artpedia":
        # Load JSON file
        data = pd.read_json(data_path, encoding='utf8', orient="index")[:data_num]
    else:  # SemArtv2
        # Load CSV file
        data = pd.read_csv(data_path, encoding='latin1', delimiter=';')[:data_num]
    
    return data


def extract_metadata(row: pd.Series, data_type: str) -> Dict[str, Any]:
    """
    Extract metadata from a row based on dataset type.

    Args:
        row: DataFrame row
        data_type: Type of dataset ("SemArtv2" or "Artpedia")

    Returns:
        dict: Metadata dictionary
    """
    if data_type == "Artpedia":
        return {
            "title": row.get('title', ''),
            "author": row.get('artists', ''),
            "year": row.get('year', ''),
            "tags": row.get('tags', ''),
        }
    else:  # SemArtv2
        return {
            "title": row.get('TITLE', ''),
            "author": row.get('AUTHOR', ''),
            "technique": row.get('TECHNIQUE', ''),
            "timeframe": row.get('TIMEFRAME', ''),
            "tags": row.get('tags', ''),
        }


def get_image_path(img_id: str, data_type: str) -> str:
    """
    Get image path based on dataset type and image ID.

    Args:
        img_id: Image ID
        data_type: Type of dataset ("SemArtv2" or "Artpedia")

    Returns:
        str: Image file path
    """
    if data_type == "Artpedia":
        return f"../../data/Artpedia/Images/{img_id}.jpg"
    else:  # SemArtv2
        return f"../../data/SemArt/Images/{img_id}"


def extract_painting_info(row: pd.Series, data_type: str) -> Dict[str, Any]:
    """
    Extract painting information from a row based on dataset type.

    Args:
        row: DataFrame row
        data_type: Type of dataset ("SemArtv2" or "Artpedia")

    Returns:
        dict: Dictionary with keys: img_id, img_path, title, author, metadata, tags
    """
    if data_type == "Artpedia":
        img_id = row.name  # Artpedia uses index as image ID
        tags = row.get('tags', '')
        artist = row.get('artists', '')
        title = row.get('title', '')
        year = row.get('year', '')
        img_path = get_image_path(img_id, data_type)
        
        metadata = {
            "title": title,
            "author": artist,
            "year": year,
            "tags": tags,
        }
        
        return {
            "img_id": img_id,
            "img_path": img_path,
            "title": title,
            "author": artist,
            "year": year,
            "tags": tags,
            "metadata": metadata,
        }
    else:  # SemArtv2
        tags = row.get('tags', '')
        author = row.get('AUTHOR', '')
        img_id = row.get('IMAGE_FILE', '')
        title = row.get('TITLE', '')
        technique = row.get('TECHNIQUE', '')
        timeframe = row.get('TIMEFRAME', '')
        img_path = get_image_path(img_id, data_type)
        
        metadata = {
            "title": title,
            "author": author,
            "technique": technique,
            "timeframe": timeframe,
            "tags": tags,
        }
        
        return {
            "img_id": img_id,
            "img_path": img_path,
            "title": title,
            "author": author,
            "technique": technique,
            "timeframe": timeframe,
            "tags": tags,
            "metadata": metadata,
        }


def run_agentic_query(
    rag: LightRAG,
    query_text: str,
    img_path: str,
    metadata: Dict[str, Any],
    mode: str,
    vlm_weight: float = 0.5,
    data_type: str = "SemArtv2",
    shot_number: int = 2,
    fewshot_type: str = "SM_fewshot",
    planner_mode: str = "full",
    **kwargs
) -> tuple[str, None, None]:
    """
    Run agentic reasoning query.

    Args:
        rag: LightRAG instance
        query_text: Query text
        img_path: Path to image file
        metadata: Metadata dictionary
        mode: Retrieval strategy mode
        vlm_weight: Weight for VLM scores in reranking (0-1)
        data_type: Dataset type ("SemArtv2" or "Artpedia")
        shot_number: Number of few-shot examples
        fewshot_type: Type of few-shot (SM_fewshot or MM_fewshot)
        planner_mode: Planning mode - "full" (default), "none", "random", "text_only"
        **kwargs: Additional query parameters (e.g., top_k)

    Returns:
        tuple: (generated_description, None, None)
    """
    multimodal_content = (
        [{"type": "image", "img_path": img_path}]
        if img_path and os.path.exists(img_path)
        else []
    )

    # Prepare kwargs for agentic query (needed for local_query context retrieval)
    query_kwargs = {
        "vlm_weight": vlm_weight,
        "data_type": data_type,
        "shot_number": shot_number,
        "fewshot_type": fewshot_type,
        "planner_mode": planner_mode,
        **kwargs
    }

    # Use synchronous wrapper if available
    if hasattr(rag, 'query_with_agentic_reasoning'):
        generated_description = rag.query_with_agentic_reasoning(
            query=query_text,
            multimodal_content=multimodal_content,
            metadata=metadata,
            mode=mode,
            **query_kwargs
        )
        return generated_description, None, None

    # Fallback to async
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            generated_description = asyncio.run(
                rag.aquery_with_agentic_reasoning(
                    query=query_text,
                    multimodal_content=multimodal_content,
                    metadata=metadata,
                    mode=mode,
                    **query_kwargs
                )
            )
        else:
            generated_description = loop.run_until_complete(
                rag.aquery_with_agentic_reasoning(
                    query=query_text,
                    multimodal_content=multimodal_content,
                    metadata=metadata,
                    mode=mode,
                    **query_kwargs
                )
            )
    except RuntimeError:
        # No event loop, create new one
        generated_description = asyncio.run(
            rag.aquery_with_agentic_reasoning(
                query=query_text,
                multimodal_content=multimodal_content,
                metadata=metadata,
                mode=mode,
                **query_kwargs
            )
        )

    return generated_description, None, None


def run_traditional_query(
    rag: LightRAG,
    query_text: str,
    img_path: str,
    args: Any
) -> tuple[str, Any, Any]:
    """
    Run traditional LightRAG query.

    Args:
        rag: LightRAG instance
        query_text: Query text
        img_path: Path to image file
        args: Arguments object

    Returns:
        tuple: (generated_description, retrieved_context, rerank_context)
    """
    query = {"text": query_text, "image": img_path}
    generated_description, retrieved_context, rerank_context = rag.query(
        query,
        param=QueryParam(mode=args.retrieval_strategy),
        data_type=args.data_type,
        shot_number=args.shot_number,
        fewshot_type=args.fewshot_type,
        vlm_weight=args.vlm_weight
    )
    return generated_description, retrieved_context, rerank_context


def run_ArtRAG_inference(
    working_dir: str,
    llm_model_func: callable,
    args: Any
) -> str:
    """
    Run inference on SemArt dataset using LightRAG.

    Supports both sequential (backward compatible) and async batch processing.

    Args:
        working_dir: Working directory for LightRAG (contains built graph)
        llm_model_func: LLM model function to use
        args: Arguments object containing inference parameters
            - use_batch_processing: If True, use async batch processing (default: False)
            - batch_size: Batch size for async processing (default: 10)
            - max_concurrent: Maximum concurrent queries (default: 5)

    Returns:
        str: Path to output file with generated descriptions
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
    
    # Log configuration clearly at start
    print("\n" + "=" * 80)
    print("ART RAG INFERENCE CONFIGURATION")
    print("=" * 80)
    print(f"Model: {model_display}")
    print(f"Query Method: {query_method}")
    if getattr(args, 'use_agentic', False):
        print(f"Planner Mode (Ablation): {planner_mode}")
    print(f"Dataset: {args.data_type}")
    print(f"Question Type: {args.question_type}")
    print(f"Shot Number: {args.shot_number}")
    print(f"Few-shot Type: {getattr(args, 'fewshot_type', 'N/A')}")
    print(f"VLM Weight: {getattr(args, 'vlm_weight', 0.5)}")
    print(f"Data Samples: {args.data_num}")
    print("=" * 80 + "\n")
    
    # Check if batch processing is enabled
    use_batch = getattr(args, 'use_batch_processing', False)
    inference_batch_size = getattr(args, 'inference_batch_size', 10)  # Different from evaluation batch_size
    max_concurrent = getattr(args, 'max_concurrent', 5)
    
    if use_batch:
        # Use async batch processing
        try:
            from .inference_utils_batch import run_ArtRAG_inference_async
        except ImportError:
            print("Warning: Batch processing module not available, falling back to sequential")
            use_batch = False
        
        if use_batch:
            # Use asyncio.run() which properly handles event loop cleanup
            # This ensures all async cleanup tasks complete before the loop closes
            try:
                return asyncio.run(
                    run_ArtRAG_inference_async(
                        working_dir, llm_model_func, args, inference_batch_size, max_concurrent
                    )
                )
            except RuntimeError as e:
                # If there's already a running event loop (e.g., in Jupyter), use nest_asyncio
                if "asyncio.run() cannot be called from a running event loop" in str(e):
                    import nest_asyncio
                    nest_asyncio.apply()
                    loop = asyncio.get_event_loop()
                    return loop.run_until_complete(
                        run_ArtRAG_inference_async(
                            working_dir, llm_model_func, args, inference_batch_size, max_concurrent
                        )
                    )
                raise
    
    # Original sequential processing (backward compatible)
    # Initialize LightRAG
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func
    )

    # Load data using unified function
    data = load_dataset(args.data_type, args.data_num)
    total_rows = len(data)
    print(f"Starting sequential inference: Processing {total_rows} rows")

    # Process each row with error handling
    results = []
    errors = []
    
    for index, row in tqdm(data.iterrows(), total=len(data), desc="Processing rows"):
        try:
            print(f"Processing row: {index}")

            # Extract painting information using unified function
            painting_info = extract_painting_info(row, args.data_type)
            img = painting_info["img_path"]
            img_id = painting_info["img_id"]
            metadata = painting_info["metadata"]

            # Validate image path
            if not os.path.exists(img):
                print(f"Warning: Image not found for row {index}: {img}, continuing without image")
                img = None

            # Build query text
            query_text = build_query_text(row, args)

            # Run inference with error handling
            try:
                if args.use_agentic:
                    generated_description, retrieved_context, rerank_context = run_agentic_query(
                        rag, query_text, img, metadata, args.retrieval_strategy,
                        vlm_weight=getattr(args, 'vlm_weight', 0.5),
                        data_type=args.data_type,
                        shot_number=args.shot_number,
                        fewshot_type=args.fewshot_type,
                        planner_mode=getattr(args, 'planner_mode', 'full')
                    )
                else:
                    generated_description, retrieved_context, rerank_context = run_traditional_query(
                        rag, query_text, img, args
                    )
            except Exception as e:
                print(f"Error: Query failed for row {index}: {e}")
                generated_description = f"ERROR: {str(e)}"
                retrieved_context = None
                rerank_context = None
                errors.append({'row': index, 'error': str(e)})

            print(f"Generated description: {generated_description[:100]}...")

            # Store result - handle different dataset structures
            result = {
                'Title': painting_info.get('title', ''),
                'Image': img_id,
                'Author': painting_info.get('author', ''),
                'Generated Description': generated_description,
                'Retrieved context': retrieved_context,
                'rerank_context': rerank_context
            }
            
            # Add dataset-specific fields
            if args.data_type == "Artpedia":
                result['Year'] = painting_info.get('year', '')
                result['Concepts'] = painting_info.get('tags', '')
            else:  # SemArtv2
                result['Technique'] = painting_info.get('technique', '')
                result['Timeframe'] = painting_info.get('timeframe', '')
                result['Concepts'] = painting_info.get('tags', '')
            
            results.append(result)
        except Exception as e:
            # Critical error - log but continue
            print(f"Critical error processing row {index}: {e}")
            painting_info = extract_painting_info(row, args.data_type)
            result = {
                'Title': painting_info.get('title', ''),
                'Image': painting_info.get('img_id', ''),
                'Author': painting_info.get('author', ''),
                'Generated Description': f"CRITICAL ERROR: {str(e)}",
                'Retrieved context': None,
                'rerank_context': None,
                'error': str(e)
            }
            
            # Add dataset-specific fields
            if args.data_type == "Artpedia":
                result['Year'] = painting_info.get('year', '')
                result['Concepts'] = painting_info.get('tags', '')
            else:  # SemArtv2
                result['Technique'] = painting_info.get('technique', '')
                result['Timeframe'] = painting_info.get('timeframe', '')
                result['Concepts'] = painting_info.get('tags', '')
            
            results.append(result)
            errors.append({'row': index, 'error': str(e)})

    # Log error summary
    if errors:
        print(f"\nWarning: Encountered {len(errors)} errors during processing")
        print(f"First few errors: {errors[:3]}")

    # Save results
    results_df = pd.DataFrame(results)
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
    # Get model name from args (it's stored as a string in args object)
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
    
    print("\n" + "=" * 80)
    print("INFERENCE COMPLETED")
    print("=" * 80)
    print(f"Model: {model_display}")
    print(f"Query Method: {query_method}")
    if getattr(args, 'use_agentic', False):
        print(f"Planner Mode (Ablation): {planner_mode}")
    print(f"Total Rows Processed: {len(data)}")
    print(f"Successful: {len(data) - len(errors)}")
    if errors:
        print(f"Errors: {len(errors)} (check 'error' field in results)")
    print(f"Output File: {output_file}")
    print("=" * 80 + "\n")
    return output_file
