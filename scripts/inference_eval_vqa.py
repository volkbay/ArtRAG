#!/usr/bin/env python
"""
Script for running ArtRAG VQA inference and evaluation on ArtCoT-QA dataset.

This script loads questions from artcot_qa_50.json, runs ArtRAG inference
to generate answers, and evaluates them against ground truth answers.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd
import numpy as np

# Load environment variables from .env file
load_dotenv(dotenv_path=".env", override=False)

# Add project root to path
sys.path.append(os.path.abspath('.'))

from artrag.llm import gpt_4o_mini_complete, gpt_4o_complete, bedrock_complete, create_bedrock_complete, bedrock_complete_if_cache
from artrag import LightRAG, QueryParam
from artrag.inference_utils import run_agentic_query, run_traditional_query
from artrag.evaluation import evaluate_batch
from artrag import clip_score
import clip
import asyncio
import torch


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run ArtRAG VQA inference and evaluation on ArtCoT-QA dataset."
    )
    
    # Model and data configuration
    parser.add_argument(
        '--working_dir', 
        type=str,
        default="./built_graph/All_gpt_4o_mini_prompt_tuning_style_event_clean",
        help='Working directory for LightRAG (contains built graph). Must be a valid path, not a placeholder.'
    )
    parser.add_argument(
        '--qa_file',
        type=str,
        default="artcot_qa_generation/artcot_qa_50.json",
        help='Path to ArtCoT-QA JSON file containing questions and answers.'
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
        help='Retrieval strategy to use (only used when --use_agentic is False).'
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
        default=None,
        help='Number of questions to process. If None, processes all questions in the dataset. Default: None (process all)'
    )
    
    # Input/Output settings
    parser.add_argument(
        '--generated_answers',
        type=str,
        help='Pre-generated answers file path (if exists, inference will be skipped)'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default=None,
        help='Directory of images. Defaults to ../../data/SemArt/Images'
    )
    
    # Processing options
    parser.add_argument(
        '--fewshot_type',
        type=str,
        default="SM_fewshot",
        choices=['MM_fewshot', 'SM_fewshot'],
        help='Type of few-shot learning examples: MultiModal (MM) or SingleModal (SM)'
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
             '"us.anthropic.claude-sonnet-4-5-20250929-v1:0"'
    )
    parser.add_argument(
        '--use_llm_judge',
        action='store_true',
        help='Enable LLM-as-a-Judge evaluation (Unified evaluation: Answer Quality, Reasoning Quality, Retrieval Quality)'
    )
    parser.add_argument(
        '--judge_model',
        type=str,
        default="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        help='Bedrock model ID to use as LLM judge. Default: us.anthropic.claude-sonnet-4-5-20250929-v1:0'
    )
    parser.add_argument(
        '--calculate_clip',
        action='store_true',
        help='Calculate CLIP scores for answer quality evaluation (deprecated, use --use_llm_judge instead)'
    )

    return parser.parse_args()


def load_qa_data(qa_file: str) -> Dict[str, Any]:
    """
    Load ArtCoT-QA data from JSON file.
    
    Args:
        qa_file: Path to JSON file containing questions and answers
        
    Returns:
        Dict containing dataset metadata and questions list
    """
    # Handle relative paths
    if not os.path.isabs(qa_file):
        script_dir = Path(__file__).parent.parent
        potential_path = script_dir / qa_file
        if potential_path.exists():
            qa_file = str(potential_path)
        elif not os.path.exists(qa_file):
            raise FileNotFoundError(f"QA file not found: {qa_file}")
    
    with open(qa_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data.get('questions', []))} questions from {qa_file}")
    return data


async def run_agentic_query_with_plans(
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
) -> tuple[str, Any, Optional[List[Dict]]]:
    """
    Run agentic reasoning query and return generation plan and context.
    
    Returns:
        tuple: (generated_answer, rerank_context, generation_plan)
    """
    from artrag.utils import validate_image_file
    from dataclasses import asdict
    from artrag.prompt_art import PROMPTS
    
    multimodal_content = (
        [{"type": "image", "img_path": img_path}]
        if img_path and os.path.exists(img_path)
        else []
    )

    # Prepare kwargs for agentic query
    query_kwargs = {
        "vlm_weight": vlm_weight,
        "data_type": data_type,
        "shot_number": shot_number,
        "fewshot_type": fewshot_type,
        "planner_mode": planner_mode,
        **kwargs
    }

    # Call agentic reasoning and capture plans
    if hasattr(rag, 'aquery_with_agentic_reasoning'):
        # Generate plans first to capture generation_plan
        query_image = None
        if multimodal_content:
            for item in multimodal_content:
                if item.get("type") == "image":
                    query_image = item.get("img_path") or item.get("image_path")
                    if query_image and validate_image_file(query_image):
                        break
        
        # Generate retrieval query from planner
        # Use VQA prompt for ArtCoT-QA question answering
        multimodal_summary = rag._summarize_multimodal_content_for_planner(multimodal_content)
        plan_prompt_key = "AGENTIC_PLAN_PROMPT_VQA"  # Use VQA-specific planner prompt
        plan_prompt = PROMPTS[plan_prompt_key].format(
            query=query_text,
            metadata=json.dumps(metadata, ensure_ascii=True),
            multimodal_summary=multimodal_summary,
        )
        
        plan_response = await rag._call_agentic_planner(plan_prompt, multimodal_content)
        plan_data = rag._parse_agentic_plan_response(plan_response)
        generation_plan = plan_data.get("generation_plan", [])
        retrieval_plan = plan_data.get("retrieval_plan", {})
        retrieval_query = retrieval_plan.get("retrieval_query", query_text)
        
        # Get context using local_query
        if query_image:
            local_query_input = {"text": retrieval_query, "image": query_image}
        else:
            local_query_input = retrieval_query
        
        query_param = QueryParam(mode = "local", only_need_context=True, **kwargs)
        from artrag.operate import local_query
        _, before_rerank_context, rerank_context = await local_query(
            local_query_input,
            rag.chunk_entity_relation_graph,
            rag.entities_vdb,
            rag.text_chunks,
            query_param,
            asdict(rag),
        )
        
        # Now call the full agentic reasoning to get the answer
        # Use VQA prompt for ArtCoT-QA question answering
        generated_answer = await rag.aquery_with_agentic_reasoning(
            query=query_text,
            multimodal_content=multimodal_content,
            metadata=metadata,
            mode=mode,
            task_type="vqa",  # Use VQA prompt for question answering
            **query_kwargs
        )
        
        return generated_answer, rerank_context, generation_plan
    
    return "", None, None


def run_vqa_inference(
    working_dir: str,
    llm_model_func: callable,
    qa_data: Dict[str, Any],
    args: Any,
    qa_file_path: str = None
) -> str:
    """
    Run VQA inference on ArtCoT-QA questions.
    
    Args:
        working_dir: Working directory for LightRAG
        llm_model_func: LLM model function to use
        qa_data: ArtCoT-QA data dictionary
        args: Arguments object containing inference parameters
        
    Returns:
        str: Path to output file with generated answers
    """
    # Validate and create working directory if needed
    if not os.path.exists(working_dir):
        if '...' in working_dir or working_dir.endswith('...'):
            # Handle placeholder paths
            print(f"Warning: Working directory appears to be a placeholder: {working_dir}")
            print("Please provide a valid working directory path.")
            raise ValueError(f"Invalid working directory: {working_dir}")
        else:
            # Try to create the directory
            try:
                os.makedirs(working_dir, exist_ok=True)
                print(f"Created working directory: {working_dir}")
            except Exception as e:
                print(f"Error creating working directory {working_dir}: {e}")
                raise
    
    # Initialize LightRAG
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func
    )
    
    questions = qa_data.get('questions', [])
    total_questions = len(questions)
    
    # Limit number of questions if data_num is specified
    if args.data_num is not None:
        questions = questions[:args.data_num]
        print(f"Processing {len(questions)} out of {total_questions} questions (limited by --data_num={args.data_num})...")
    else:
        print(f"Processing all {total_questions} questions...")
    
    results = []
    errors = []
    
    for idx, qa_item in enumerate(tqdm(questions, desc="Processing questions")):
        try:
            question = qa_item['question']
            ground_truth_answer = qa_item['answer']
            painting_id = qa_item.get('painting_id', '')
            image_path = qa_item.get('image_path', '')
            metadata = qa_item.get('metadata', {})
            
            # Handle relative image paths
            if image_path and not os.path.isabs(image_path):
                # Try relative to QA file directory first (since paths in JSON are relative to that)
                if qa_file_path:
                    qa_file_dir = Path(qa_file_path).parent
                    if not os.path.isabs(qa_file_path):
                        # Resolve relative to script directory
                        script_dir = Path(__file__).parent.parent
                        qa_file_dir = script_dir / Path(qa_file_path).parent
                    
                    potential_path = qa_file_dir / image_path
                    if potential_path.exists():
                        image_path = str(potential_path.resolve())
                
                # If still not found, try relative to script directory
                if not os.path.exists(image_path) if image_path else True:
                    script_dir = Path(__file__).parent.parent
                    potential_path = script_dir / image_path
                    if potential_path.exists():
                        image_path = str(potential_path.resolve())
                    elif args.image_dir:
                        # Try image_dir
                        potential_path = Path(args.image_dir) / os.path.basename(image_path)
                        if potential_path.exists():
                            image_path = str(potential_path.resolve())
            
            # Validate image path
            if image_path and not os.path.exists(image_path):
                print(f"Warning: Image not found for question {idx}: {image_path}")
                image_path = None
            
            # Use question as query text
            query_text = question
            
            # Run inference
            try:
                if args.use_agentic:
                    # Use async version to capture plans
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import nest_asyncio
                        nest_asyncio.apply()
                        generated_answer, rerank_context, generation_plan = asyncio.run(
                            run_agentic_query_with_plans(
                                rag, query_text, image_path, metadata, args.retrieval_strategy,
                                vlm_weight=args.vlm_weight,
                                data_type=qa_data.get('dataset_type', 'SemArtv2'),
                                shot_number=args.shot_number,
                                fewshot_type=args.fewshot_type,
                                planner_mode=args.planner_mode
                            )
                        )
                    else:
                        generated_answer, rerank_context, generation_plan = loop.run_until_complete(
                            run_agentic_query_with_plans(
                                rag, query_text, image_path, metadata, args.retrieval_strategy,
                                vlm_weight=args.vlm_weight,
                                data_type=qa_data.get('dataset_type', 'SemArtv2'),
                                shot_number=args.shot_number,
                                fewshot_type=args.fewshot_type,
                                planner_mode=args.planner_mode
                            )
                        )
                else:
                    # Use traditional query method
                    # Create a mock args object for run_traditional_query
                    class MockArgs:
                        def __init__(self):
                            self.retrieval_strategy = args.retrieval_strategy
                            self.data_type = qa_data.get('dataset_type', 'SemArtv2')
                            self.shot_number = args.shot_number
                            self.fewshot_type = args.fewshot_type
                            self.vlm_weight = args.vlm_weight
                    
                    mock_args = MockArgs()
                    _, _, rerank_context = run_traditional_query(
                        rag, query_text, image_path, mock_args
                    )
                    generation_plan = None
            except Exception as e:
                print(f"Error: Query failed for question {idx}: {e}")
                generated_answer = f"ERROR: {str(e)}"
                rerank_context = None
                generation_plan = None
                errors.append({'question_idx': idx, 'error': str(e)})
            
            # Store result
            result = {
                'question_idx': idx,
                'question': question,
                'ground_truth_answer': ground_truth_answer,
                'generated_answer': generated_answer,
                'painting_id': painting_id,
                'image_path': image_path,
                'rerank_context': rerank_context,
                'generation_plan': generation_plan,  # Add generation plan
                'metadata': metadata,
                'cot_steps': qa_item.get('cot_steps', []),
                'evidence_types': qa_item.get('evidence_types', []),
                'difficulty': qa_item.get('difficulty', 'unknown'),
                'planning_complexity': qa_item.get('planning_complexity', 'unknown')
            }
            results.append(result)
            
        except Exception as e:
            print(f"Critical error processing question {idx}: {e}")
            errors.append({'question_idx': idx, 'error': str(e)})
            results.append({
                'question_idx': idx,
                'question': qa_item.get('question', ''),
                'ground_truth_answer': qa_item.get('answer', ''),
                'generated_answer': f"CRITICAL ERROR: {str(e)}",
                'error': str(e)
            })
    
    # Log error summary
    if errors:
        print(f"\nWarning: Encountered {len(errors)} errors during processing")
        print(f"First few errors: {errors[:3]}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        working_dir,
        f"output_vqa_{timestamp}"
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
        f'generated_answers_{query_method}_{model_suffix}_{timestamp}.json'
    )
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved results to: {output_file}")
    return output_file


async def llm_judge_unified_evaluation(
    question: str,
    ground_truth_answer: str,
    ground_truth_cot_steps: List[Dict],
    generated_answer: str,
    rerank_context: Any,
    generation_plan: Optional[List[Dict]],
    judge_model: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
) -> Dict[str, Any]:
    """
    Unified LLM-as-a-Judge evaluation for ArtCoT-QA.
    
    Evaluates all metrics in a single call:
    - Reasoning Quality: CoT Faithfulness
    - Retrieval Quality: Subgraph Relevance, Evidence Coverage
    - Generation Quality: Answer Quality
    
    Args:
        question: The question being answered
        ground_truth_answer: Ground truth answer
        ground_truth_cot_steps: Ground truth CoT steps with grounding tags
        generated_answer: Generated answer from the system
        rerank_context: Reranked context subgraph
        generation_plan: Generation plan/reasoning steps from agentic system (if available)
        judge_model: Model to use as judge
    
    Returns:
        dict with all evaluation scores
    """
    # Format ground truth CoT steps
    gt_cot_text = "\n".join([
        f"Step {i+1} [{step.get('grounding', 'Unknown')}]: {step.get('step', '')}"
        for i, step in enumerate(ground_truth_cot_steps)
    ])
    
    # Extract expected evidence types from ground truth CoT
    expected_evidence_types = set()
    for step in ground_truth_cot_steps:
        grounding = step.get('grounding', '')
        if '[Visual]' in grounding:
            expected_evidence_types.add('Visual')
        if '[Description]' in grounding:
            expected_evidence_types.add('Description')
        if '[Metadata]' in grounding:
            expected_evidence_types.add('Metadata')
        if '[KG-Background]' in grounding:
            expected_evidence_types.add('KG-Background')
        if '[Common-Knowledge]' in grounding:
            expected_evidence_types.add('Common-Knowledge')
    
    # Format reranked context
    context_text = ""
    if rerank_context:
        if isinstance(rerank_context, list):
            context_text = "\n".join([str(item) for item in rerank_context[:10]])  # Limit to top 10
        else:
            context_text = str(rerank_context)
        # Limit total length
        context_text = context_text[:3000] if len(context_text) > 3000 else context_text
    else:
        context_text = "None (no context retrieved)"
    
    # Format generation plan / reasoning steps
    reasoning_text = ""
    if generation_plan:
        reasoning_text = "\n".join([
            f"Step {i+1}: {step.get('goal', step.get('step', str(step)))} [Evidence: {step.get('evidence', 'unknown')}]"
            for i, step in enumerate(generation_plan)
        ])
    else:
        reasoning_text = "Not available (system did not provide explicit reasoning steps)"
    
    prompt = f"""You are an expert art historian evaluating a visual question answering system on ArtCoT-QA.

QUESTION: {question}

GROUND TRUTH ANSWER: {ground_truth_answer}

GROUND TRUTH REASONING STEPS (with evidence types):
{gt_cot_text}

EXPECTED EVIDENCE TYPES: {', '.join(sorted(expected_evidence_types)) if expected_evidence_types else 'None specified'}

---
SYSTEM OUTPUT TO EVALUATE:
---

GENERATED ANSWER:
{generated_answer}

SYSTEM'S REASONING PLAN/STEPS:
{reasoning_text}

RERANKED CONTEXT SUBGRAPH (retrieved knowledge):
{context_text}

---
EVALUATION TASK:
---

Evaluate the system's performance across five dimensions using a 1-5 scale:

1. **CoT Faithfulness** (Reasoning Quality): Does the generated answer follow a faithful chain-of-thought reasoning process? Is it grounded in evidence and avoid hallucination?
   - 1: Major hallucinations, not grounded, no clear reasoning chain
   - 2: Some hallucinations, weakly grounded, unclear reasoning
   - 3: Mostly faithful with minor speculation, some reasoning steps visible
   - 4: Faithful with good grounding, clear multi-step reasoning
   - 5: Completely faithful, strongly grounded, excellent multi-step reasoning chain

2. **CoT Step Completeness** (Reasoning Quality): Does the system's reasoning plan cover all the necessary reasoning steps compared to the ground truth CoT steps? Are all key reasoning components present?
   - 1: Missing most reasoning steps (0-1 steps covered)
   - 2: Missing several reasoning steps (1-2 steps covered)
   - 3: Some reasoning steps covered (2-3 steps covered)
   - 4: Most reasoning steps covered (3-4 steps covered)
   - 5: All necessary reasoning steps covered (4-5 steps covered, complete reasoning chain)

3. **Subgraph Relevance** (Retrieval Quality): Is the retrieved context subgraph relevant to the question and ground truth reasoning steps?
   - 1: Completely irrelevant subgraph
   - 2: Mostly irrelevant with few relevant parts
   - 3: Partially relevant, some useful information
   - 4: Mostly relevant, good coverage of needed information
   - 5: Highly relevant, directly addresses question and reasoning needs

4. **Evidence Coverage** (Retrieval Quality): Does the retrieved context cover the expected evidence types needed for the multi-step reasoning?
   - 1: Missing most necessary evidence types (0-1 types covered)
   - 2: Missing several evidence types (1-2 types covered)
   - 3: Some evidence types covered (2-3 types covered)
   - 4: Most evidence types covered (3-4 types covered)
   - 5: All expected evidence types covered (4-5 types covered)

5. **Answer Quality** (Generation Quality): Is the generated answer correct, meaningful, and informative?
   - 1: Completely incorrect or irrelevant
   - 2: Mostly incorrect with some relevant points
   - 3: Partially correct but missing key elements
   - 4: Mostly correct with minor inaccuracies
   - 5: Completely correct, accurate, and informative

Provide your evaluation as JSON with this structure:
{{
    "cot_faithfulness_score": <integer 1-5>,
    "cot_step_completeness_score": <integer 1-5>,
    "subgraph_relevance_score": <integer 1-5>,
    "evidence_coverage_score": <integer 1-5>,
    "answer_quality_score": <integer 1-5>,
    "cot_faithfulness_reasoning": "<brief explanation>",
    "cot_step_completeness_reasoning": "<brief explanation>",
    "subgraph_relevance_reasoning": "<brief explanation>",
    "evidence_coverage_reasoning": "<brief explanation>",
    "answer_quality_reasoning": "<brief explanation>",
    "overall_assessment": "<brief overall assessment>"
}}

Return ONLY valid JSON, no other text."""

    try:
        response = await bedrock_complete_if_cache(
            model=judge_model,
            prompt=prompt,
            system_prompt="You are an expert art historian evaluating VQA systems. Return only valid JSON."
        )
        # Parse JSON response
        response_cleaned = response.strip()
        if "```json" in response_cleaned:
            response_cleaned = response_cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in response_cleaned:
            response_cleaned = response_cleaned.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_cleaned)
        return result
    except Exception as e:
        print(f"Warning: LLM judge unified evaluation failed: {e}")
        return {
            "cot_faithfulness_score": 1,
            "cot_step_completeness_score": 1,
            "subgraph_relevance_score": 1,
            "evidence_coverage_score": 1,
            "answer_quality_score": 1,
            "error": str(e)
        }


async def evaluate_vqa_with_unified_llm_judge(
    results: List[Dict],
    judge_model: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
) -> Dict[str, Any]:
    """
    Evaluate VQA results using unified LLM-as-a-Judge evaluation.
    
    Returns:
        dict with evaluation scores for all metrics
    """
    all_scores = []
    
    print("\nRunning Unified LLM-as-a-Judge evaluation...")
    
    for result in tqdm(results, desc="LLM Judge Evaluation"):
        question = result.get('question', '')
        ground_truth_answer = result.get('ground_truth_answer', '')
        generated_answer = result.get('generated_answer', '')
        cot_steps = result.get('cot_steps', [])
        rerank_context = result.get('rerank_context')
        generation_plan = result.get('generation_plan')
        
        if not generated_answer or generated_answer.startswith('ERROR'):
            continue
        
        # Unified evaluation
        evaluation = await llm_judge_unified_evaluation(
            question=question,
            ground_truth_answer=ground_truth_answer,
            ground_truth_cot_steps=cot_steps,
            generated_answer=generated_answer,
            rerank_context=rerank_context,
            generation_plan=generation_plan,
            judge_model=judge_model
        )
        all_scores.append(evaluation)
    
    # Aggregate scores
    if not all_scores:
        return {
            "error": "No valid evaluations completed"
        }
    
    eval_results = {
        'reasoning_quality': {
            'mean_cot_faithfulness': np.mean([s.get('cot_faithfulness_score', 1) for s in all_scores]),
            'mean_cot_step_completeness': np.mean([s.get('cot_step_completeness_score', 1) for s in all_scores]),
            'cot_faithfulness_scores': [s.get('cot_faithfulness_score', 1) for s in all_scores],
            'cot_step_completeness_scores': [s.get('cot_step_completeness_score', 1) for s in all_scores],
            'scale': '1-5'
        },
        'retrieval_quality': {
            'mean_subgraph_relevance': np.mean([s.get('subgraph_relevance_score', 1) for s in all_scores]),
            'mean_evidence_coverage': np.mean([s.get('evidence_coverage_score', 1) for s in all_scores]),
            'subgraph_relevance_scores': [s.get('subgraph_relevance_score', 1) for s in all_scores],
            'evidence_coverage_scores': [s.get('evidence_coverage_score', 1) for s in all_scores],
            'scale': '1-5'
        },
        'generation_quality': {
            'mean_answer_quality': np.mean([s.get('answer_quality_score', 1) for s in all_scores]),
            'scores': [s.get('answer_quality_score', 1) for s in all_scores],
            'scale': '1-5'
        },
        'all_evaluations': all_scores,
        'total_evaluated': len(all_scores)
    }
    
    return eval_results


def calculate_clip_scores_for_answers(
    results: List[Dict],
    image_dir: str = None
) -> Dict[int, float]:
    """
    Calculate CLIP scores for generated answers.
    
    Args:
        results: List of result dictionaries with 'image_path' and 'generated_answer'
        image_dir: Optional image directory (if image_paths are relative)
    
    Returns:
        dict mapping question_idx to CLIP score
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, _ = clip.load("ViT-B/32", device=device, jit=False)
        model.eval()
    except Exception as e:
        print(f"Warning: Failed to load CLIP model: {e}")
        return {}
    
    clip_scores = {}
    valid_results = []
    valid_indices = []
    
    for result in results:
        image_path = result.get('image_path', '')
        generated_answer = result.get('generated_answer', '')
        
        if not image_path or not generated_answer:
            continue
        
        if generated_answer.startswith('ERROR') or generated_answer.startswith('CRITICAL ERROR'):
            continue
        
        # Resolve image path
        if not os.path.isabs(image_path):
            if image_dir:
                potential_path = Path(image_dir) / os.path.basename(image_path)
                if potential_path.exists():
                    image_path = str(potential_path)
            if not os.path.exists(image_path):
                continue
        
        if os.path.exists(image_path):
            valid_results.append({
                'image_path': image_path,
                'answer': generated_answer,
                'idx': result.get('question_idx', 0)
            })
            valid_indices.append(result.get('question_idx', 0))
    
    if not valid_results:
        print("Warning: No valid image-answer pairs for CLIP score calculation")
        return {}
    
    try:
        image_paths = [r['image_path'] for r in valid_results]
        answers = [r['answer'] for r in valid_results]
        
        image_feats = clip_score.extract_all_images(
            image_paths, model, device, batch_size=64, num_workers=8
        )
        
        _, per_instance_scores, _ = clip_score.get_clip_score(
            model, image_feats, answers, device
        )
        
        for idx, score in zip(valid_indices, per_instance_scores):
            clip_scores[idx] = float(score)
        
        mean_clip = np.mean(list(clip_scores.values()))
        print(f"Mean CLIP Score: {mean_clip:.4f}")
        
    except Exception as e:
        print(f"Warning: CLIP score calculation failed: {e}")
    
    return clip_scores


def evaluate_vqa_answers(
    generated_answers_file: str,
    args: Any
) -> None:
    """
    Evaluate generated VQA answers against ground truth.
    
    Args:
        generated_answers_file: Path to JSON file with generated answers
        args: Arguments object
    """
    # Load generated answers
    with open(generated_answers_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Prepare lists for evaluation
    predicts = []
    answers = []
    
    for result in results:
        if 'generated_answer' in result and 'ground_truth_answer' in result:
            # Clean answers (remove error messages)
            gen_answer = result['generated_answer']
            gt_answer = result['ground_truth_answer']
            
            if not gen_answer.startswith('ERROR') and not gen_answer.startswith('CRITICAL ERROR'):
                predicts.append(gen_answer)
                answers.append([gt_answer])  # Wrap in list for evaluation format
    
    if not predicts:
        print("Error: No valid predictions to evaluate")
        return
    
    # Skip mscoco evaluation for ArtCoT-QA (question-answering task)
    # ArtCoT-QA uses LLM-as-a-Judge evaluation instead
    print(f"\nSkipping mscoco evaluation for ArtCoT-QA (question-answering task).")
    print(f"Using LLM-as-a-Judge evaluation instead for {len(predicts)} question-answer pairs.")
    
    # Initialize evaluation results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_output_file = generated_answers_file.replace('.json', f'_eval_{timestamp}.json')
    
    eval_results = {
        'total_questions': len(predicts),
        'evaluation_type': 'ArtCoT-QA (LLM-as-a-Judge only)',
        'note': 'mscoco metrics skipped for question-answering task'
    }
    
    # Unified LLM-as-a-Judge Evaluation
    if args.use_llm_judge:
        print("\n" + "=" * 80)
        print("UNIFIED LLM-AS-A-JUDGE EVALUATION")
        print("=" * 80)
        
        try:
            llm_judge_results = asyncio.run(
                evaluate_vqa_with_unified_llm_judge(results, judge_model=args.judge_model)
            )
            eval_results['llm_judge'] = llm_judge_results
            
            # Print LLM judge results
            if 'error' not in llm_judge_results:
                print("\nReasoning Quality (Scale: 1-5):")
                print(f"  Mean CoT Faithfulness: {llm_judge_results['reasoning_quality']['mean_cot_faithfulness']:.2f}/5.0")
                print(f"  Mean CoT Step Completeness: {llm_judge_results['reasoning_quality']['mean_cot_step_completeness']:.2f}/5.0")
                
                print("\nRetrieval Quality (Scale: 1-5):")
                print(f"  Mean Subgraph Relevance: {llm_judge_results['retrieval_quality']['mean_subgraph_relevance']:.2f}/5.0")
                print(f"  Mean Evidence Coverage: {llm_judge_results['retrieval_quality']['mean_evidence_coverage']:.2f}/5.0")
                
                print("\nGeneration Quality (Scale: 1-5):")
                print(f"  Mean Answer Quality: {llm_judge_results['generation_quality']['mean_answer_quality']:.2f}/5.0")
                
                print(f"\nTotal Evaluated: {llm_judge_results['total_evaluated']}")
            else:
                print(f"Error in LLM-as-a-Judge evaluation: {llm_judge_results.get('error', 'Unknown error')}")
            
        except Exception as e:
            print(f"Error in LLM-as-a-Judge evaluation: {e}")
            eval_results['llm_judge_error'] = str(e)
    
    # CLIP Score Calculation (deprecated, kept for backward compatibility)
    if args.calculate_clip:
        print("\n" + "=" * 80)
        print("CLIP SCORE CALCULATION (Deprecated - use --use_llm_judge instead)")
        print("=" * 80)
        
        try:
            # Determine image directory
            image_dir = args.image_dir
            if not image_dir:
                # Try to infer from first result
                first_result = results[0] if results else {}
                image_path = first_result.get('image_path', '')
                if image_path:
                    # Extract directory from image path
                    if os.path.exists(image_path):
                        image_dir = os.path.dirname(image_path)
                    elif '../../data/SemArt/Images' in image_path:
                        script_dir = Path(__file__).parent.parent
                        image_dir = str(script_dir / '../../data/SemArt/Images')
            
            clip_scores = calculate_clip_scores_for_answers(results, image_dir=image_dir)
            eval_results['clip_scores'] = {
                'per_question': clip_scores,
                'mean_score': np.mean(list(clip_scores.values())) if clip_scores else 0.0
            }
            print(f"Mean CLIP Score: {eval_results['clip_scores']['mean_score']:.4f}")
            
        except Exception as e:
            print(f"Error in CLIP score calculation: {e}")
            eval_results['clip_score_error'] = str(e)
    
    # Save evaluation results
    with open(eval_output_file, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved evaluation results to: {eval_output_file}")


def main():
    """Main execution function."""
    args = parse_arguments()
    print(f"Arguments: {args}")
    
    # Map llm_model_func argument to actual function
    if args.llm_model_func == 'bedrock_complete':
        llm_model_func = create_bedrock_complete(args.bedrock_model)
    else:
        llm_model_func_map = {
            'gpt_4o_mini_complete': gpt_4o_mini_complete,
            'gpt_4o_complete': gpt_4o_complete,
        }
        llm_model_func = llm_model_func_map[args.llm_model_func]
    
    # Load QA data
    qa_data = load_qa_data(args.qa_file)
    
    # Run inference or use pre-generated file
    if not args.generated_answers:
        generated_answers_file = run_vqa_inference(
            args.working_dir,
            llm_model_func,
            qa_data,
            args,
            qa_file_path=args.qa_file
        )
    else:
        generated_answers_file = args.generated_answers
    
    # Evaluate answers
    print(f"\nEvaluating answers from: {generated_answers_file}")
    evaluate_vqa_answers(generated_answers_file, args)


if __name__ == "__main__":
    main()
