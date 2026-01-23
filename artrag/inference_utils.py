"""
Inference utilities for ArtRAG

Contains functions for running inference on datasets, building queries,
and handling both traditional and agentic reasoning queries.
"""

import os
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import pandas as pd
import json

from .lightrag import LightRAG, QueryParam

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
current_date = datetime.now().strftime("%Y-%m-%d")


def build_query_text(row: pd.Series, args: Any) -> str:
    """
    Build query text based on question type and available data fields.

    Args:
        row: DataFrame row containing painting data
        args: Arguments object with question_type and data_type

    Returns:
        str: Formatted query text
    """
    # Determine which fields exist
    content = row.get('content', '')
    context = row.get('context', '')
    form = row.get('form', '') if args.data_type == "SemArtv2" else None

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


def get_semart_data_path(data_type: str) -> str:
    """
    Get the path to SemArt dataset file based on data type.

    Args:
        data_type: Type of SemArt dataset

    Returns:
        str: Path to dataset CSV file
    """
    semartv1_types = {"SemArtv1-content", "SemArtv1-context"}
    if data_type in semartv1_types:
        return "../../data/SemArt/semartv1_test_overlap_with_captions.csv"
    elif data_type == "SemArtv2":
        return "../../data/SemArt/semartv2_test_overlap_with_captions.csv"
    else:
        raise ValueError(f"Unknown data_type: {data_type}")


def run_agentic_query(
    rag: LightRAG,
    query_text: str,
    img_path: str,
    metadata: Dict[str, Any],
    mode: str
) -> tuple[str, None, None]:
    """
    Run agentic reasoning query.

    Args:
        rag: LightRAG instance
        query_text: Query text
        img_path: Path to image file
        metadata: Metadata dictionary
        mode: Retrieval strategy mode

    Returns:
        tuple: (generated_description, None, None)
    """
    multimodal_content = (
        [{"type": "image", "img_path": img_path}]
        if img_path and os.path.exists(img_path)
        else []
    )

    # Use synchronous wrapper if available
    if hasattr(rag, 'query_with_agentic_reasoning'):
        generated_description = rag.query_with_agentic_reasoning(
            query=query_text,
            multimodal_content=multimodal_content,
            metadata=metadata,
            mode=mode,
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
                )
            )
        else:
            generated_description = loop.run_until_complete(
                rag.aquery_with_agentic_reasoning(
                    query=query_text,
                    multimodal_content=multimodal_content,
                    metadata=metadata,
                    mode=mode,
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

    Args:
        working_dir: Working directory for LightRAG (contains built graph)
        llm_model_func: LLM model function to use
        args: Arguments object containing inference parameters

    Returns:
        str: Path to output file with generated descriptions
    """
    # Initialize LightRAG
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func
    )

    # Get dataset path
    directory = get_semart_data_path(args.data_type)
    print(f"Dataset type: {args.data_type}, question type: {args.question_type}")

    # Load data
    data = pd.read_csv(directory, encoding='latin1', delimiter=';')[:args.data_num]

    # Process each row
    results = []
    for index, row in tqdm(data.iterrows(), total=len(data), desc="Processing rows"):
        print(f"Processing row: {index}")

        # Extract painting information
        tags = row.get('tags', '')
        author = row.get('AUTHOR', '')
        img_id = row.get('IMAGE_FILE', '')
        title = row.get('TITLE', '')
        technique = row.get('TECHNIQUE', '')
        timeframe = row.get('TIMEFRAME', '')
        img = f"../../data/SemArt/Images/{img_id}"

        # Build query text
        query_text = build_query_text(row, args)

        # Prepare metadata for agentic queries
        metadata = {
            "title": title,
            "author": author,
            "technique": technique,
            "timeframe": timeframe,
            "tags": tags,
        }

        # Run inference
        if args.use_agentic:
            generated_description, retrieved_context, rerank_context = run_agentic_query(
                rag, query_text, img, metadata, args.retrieval_strategy
            )
        else:
            generated_description, retrieved_context, rerank_context = run_traditional_query(
                rag, query_text, img, args
            )

        print(f"Generated description: {generated_description[:100]}...")

        # Store result
        results.append({
            'Title': title,
            'Image': img_id,
            'Author': author,
            'Technique': technique,
            'Timeframe': timeframe,
            "Concepts": tags,
            'Generated Description': generated_description,
            'Retrieved context': retrieved_context,
            'rerank_context': rerank_context
        })

    # Save results
    results_df = pd.DataFrame(results)
    output_dir = os.path.join(
        working_dir,
        f"output_{current_date}_{args.data_type}_{args.data_num}data"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Generate output filename
    query_method = "agentic" if args.use_agentic else args.retrieval_strategy
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

    print(f"Inference completed. Generated descriptions saved to '{output_file}'.")
    return output_file
