import base64
import copy
import json
import os
import pdb
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import aioboto3
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from torchvision.transforms.functional import InterpolationMode

from .base import BaseKVStorage
from .utils import (compute_args_hash, log_gpu_memory, logger,
                    reset_gpu_peak, wrap_embedding_func_with_attrs)
from .runtime_config import settings

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    # Load .env file if it exists (won't override existing env vars)
    load_dotenv(dotenv_path=".env", override=False)
    # Also try loading from parent directory (in case script is in subdirectory)
    load_dotenv(dotenv_path="../.env", override=False)
except ImportError:
    pass  # python-dotenv not installed, rely on environment variables

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
    Timeout,
)
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _normalize_local_model_name(model_name: str) -> str:
    """Resolve local model paths so cache keys stay stable across aliases."""
    try:
        model_path = Path(model_name)
        if model_name.startswith(".") or model_path.exists():
            return str(model_path.expanduser().resolve())
    except Exception:
        pass
    return model_name


def _is_local_model_name(model_name: str) -> bool:
    try:
        model_path = Path(model_name)
        return model_name.startswith(".") or model_path.exists()
    except Exception:
        return False

def encode_image(image_path: str) -> str:
    from .utils import resize_image_if_large

    usable_path = resize_image_if_large(image_path)
    with open(usable_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def openai_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    base_url=None,
    api_key=None,
    query_image_path: Optional[str] = None,
    system_image_paths: Optional[str] = None,
    **kwargs,
) -> str:
    # Use provided api_key, or fall back to environment variable
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    elif not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY not found. Please set it in environment variables or .env file, "
            "or pass it as api_key parameter."
        )

    openai_async_client = (
        AsyncOpenAI() if base_url is None else AsyncOpenAI(base_url=base_url)
    )
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    # pdb.set_trace()
    if system_prompt:
        system_message = {"role": "user", "content": system_prompt}
        if system_image_paths:  # Assuming system_image_paths is a list of image paths
            images_content = []
            for image_path in system_image_paths:
                base64_image = encode_image(image_path)
                images_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}})
            
            system_message["content"] = [
                {"type": "text", "text": system_prompt},
                *images_content  # Unpack the list of image contents
            ]
        messages.append(system_message)
    messages.extend(history_messages)
    user_message = {"role": "user", "content": prompt}
    """
    To be done on image-text pair incontext learning
    """
    if query_image_path:
        base64_image = encode_image(query_image_path)
        user_message["content"] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png; base64, {base64_image}"}}
        ]
    
    messages.append(user_message)

    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
    return response.choices[0].message.content


class BedrockError(Exception):
    """Generic error for issues related to Amazon Bedrock"""


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, max=60),
    retry=retry_if_exception_type((BedrockError)),
)
async def bedrock_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    aws_access_key_id=None,
    aws_secret_access_key=None,
    aws_session_token=None,
    aws_region=None,
    query_image_path: Optional[str] = None,
    system_image_paths: Optional[List[str]] = None,
    **kwargs,
) -> str:
    # Only set environment variables if they are provided and not None
    # If not provided, boto3 will use default credentials (IAM role, ~/.aws/credentials, etc.)
    if aws_access_key_id is not None:
        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
    if aws_secret_access_key is not None:
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
    if aws_session_token is not None:
        os.environ["AWS_SESSION_TOKEN"] = aws_session_token
    
    # Get AWS region from parameter, environment variable, or use default
    region = aws_region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"

    # Remove image parameters from kwargs to avoid passing them to Bedrock API
    kwargs.pop("query_image_path", None)
    kwargs.pop("system_image_paths", None)

    # Fix message history format
    messages = []
    for history_message in history_messages:
        message = copy.copy(history_message)
        # Handle both text and multimodal content
        if isinstance(message.get("content"), list):
            # Already in multimodal format
            messages.append(message)
        else:
            # Convert text to format
            message["content"] = [{"text": message["content"]}]
            messages.append(message)

    # Build user message content
    user_content = [{"text": prompt}]
    
    # Add query image if provided
    if query_image_path:
        from .utils import encode_image_to_base64, validate_image_file
        if validate_image_file(query_image_path):
            image_base64 = encode_image_to_base64(query_image_path)
            if image_base64:
                # Detect image format from file extension
                img_ext = Path(query_image_path).suffix.lower()
                format_map = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", 
                             ".gif": "gif", ".webp": "webp", ".bmp": "bmp"}
                img_format = format_map.get(img_ext, "jpeg")  # Default to jpeg
                
                user_content.append({
                    "image": {
                        "format": img_format,
                        "source": {
                            "bytes": base64.b64decode(image_base64)
                        }
                    }
                })

    # Add user message
    messages.append({"role": "user", "content": user_content})
    
    # Handle system images if provided
    system_content = None
    if system_prompt:
        system_content = [{"text": system_prompt}]
        if system_image_paths:
            from .utils import encode_image_to_base64, validate_image_file
            for img_path in system_image_paths:
                if validate_image_file(img_path):
                    image_base64 = encode_image_to_base64(img_path)
                    if image_base64:
                        # Detect image format from file extension
                        img_ext = Path(img_path).suffix.lower()
                        format_map = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", 
                                     ".gif": "gif", ".webp": "webp", ".bmp": "bmp"}
                        img_format = format_map.get(img_ext, "jpeg")  # Default to jpeg
                        
                        system_content.append({
                            "image": {
                                "format": img_format,
                                "source": {
                                    "bytes": base64.b64decode(image_base64)
                                }
                            }
                        })

    # Initialize Converse API arguments
    args = {"modelId": model, "messages": messages}

    # Define system prompt (with images if provided)
    if system_content:
        args["system"] = system_content

    # Map and set up inference parameters
    inference_params_map = {
        "max_tokens": "maxTokens",
        "top_p": "topP",
        "stop_sequences": "stopSequences",
    }
    if inference_params := list(
        set(kwargs) & set(["max_tokens", "temperature", "top_p", "stop_sequences"])
    ):
        args["inferenceConfig"] = {}
        for param in inference_params:
            args["inferenceConfig"][inference_params_map.get(param, param)] = (
                kwargs.pop(param)
            )

    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    # Call model via Converse API
    session = aioboto3.Session()
    async with session.client("bedrock-runtime", region_name=region) as bedrock_async_client:
        try:
            response = await bedrock_async_client.converse(**args, **kwargs)
        except Exception as e:
            raise BedrockError(e)

        if hashing_kv is not None:
            await hashing_kv.upsert(
                {
                    args_hash: {
                        "return": response["output"]["message"]["content"][0]["text"],
                        "model": model,
                    }
                }
            )

        return response["output"]["message"]["content"][0]["text"]


@lru_cache(maxsize=1)
def initialize_hf_model(model_name):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    hf_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=True,
    )

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)

    # InternVL3 is loaded via trust_remote_code, so its model class
    # (InternVLChatModel) is NOT in AutoModelForCausalLM's static mapping --
    # it must be resolved through config.auto_map instead.
    class_ref = config.auto_map["AutoModelForCausalLM"]
    model_class = get_class_from_dynamic_module(class_ref, model_name, local_files_only=True)

    # Patch ONLY InternVL3's specific dynamic class -- never the shared
    # PreTrainedModel base, and never AutoModelForCausalLM itself -- so
    # BART and every other model loaded later in the same process keep
    # their normal, correct tied-weights resolution untouched.
    if not hasattr(model_class, '_all_tied_weights_keys_patch_applied'):
        @property
        def all_tied_weights_keys(self):
            if hasattr(self, '_tied_weights_keys'):
                return dict(self._tied_weights_keys) if self._tied_weights_keys else {}
            return {}

        @all_tied_weights_keys.setter
        def all_tied_weights_keys(self, value):
            pass

        model_class.all_tied_weights_keys = all_tied_weights_keys
        model_class._all_tied_weights_keys_patch_applied = True

    # Attention kernel selection. InternVL3's modeling code only knows
    # flash_attention_2 (when flash-attn is installed) vs. eager, and hard-codes
    # the LLM to *eager* otherwise -- the memory-hungry path that materializes the
    # full heads x seq^2 score matrix and OOMs at large rerank top_k. We pass
    # use_flash_attn only for the flash path, then promote the LLM to SDPA (a
    # drop-in, math-equivalent, memory-efficient kernel) when SDPA is requested.
    attn_impl = getattr(settings, "attn_implementation", "sdpa")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=None,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=False,
        use_flash_attn=(attn_impl == "flash_attention_2"),
    )
    if attn_impl != "flash_attention_2":
        lm = getattr(hf_model, "language_model", None)
        if lm is not None:
            try:
                lm.set_attn_implementation(attn_impl)   # transformers >= 4.48
            except Exception:
                lm.config._attn_implementation = attn_impl

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_model = hf_model.to(device)
    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token
    return hf_model, hf_tokenizer


@lru_cache(maxsize=2)
def initialize_hf_text_encoder(model_name, device: str = "cpu"):
    local_files_only = _is_local_model_name(model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    text_encoder = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
        dtype=torch.float16 if device != "cpu" and torch.cuda.is_available() else None,
    )

    text_encoder.to(device)
    text_encoder.eval()
    log_gpu_memory("after bge-m3 load")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return text_encoder, tokenizer


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def hf_model_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    **kwargs,
) -> str:
    model_name = _normalize_local_model_name(model)
    hf_model, hf_tokenizer = initialize_hf_model(model_name)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    kwargs.pop("hashing_kv", None)
    input_prompt = ""
    try:
        input_prompt = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        try:
            ori_message = copy.deepcopy(messages)
            if messages[0]["role"] == "system":
                messages[1]["content"] = (
                    "<system>"
                    + messages[0]["content"]
                    + "</system>\n"
                    + messages[1]["content"]
                )
                messages = messages[1:]
                input_prompt = hf_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        except Exception:
            len_message = len(ori_message)
            for msgid in range(len_message):
                input_prompt = (
                    input_prompt
                    + "<"
                    + ori_message[msgid]["role"]
                    + ">"
                    + ori_message[msgid]["content"]
                    + "</"
                    + ori_message[msgid]["role"]
                    + ">\n"
                )

    # Special handling for vision-language models (e.g., InternVLChatModel).
    # If the model expects img_context_token_id but it's None, return a fallback message.
    if hasattr(hf_model, "img_context_token_id") and hf_model.img_context_token_id is None:
        return (
            "Note: This is a vision-language model. For best results, provide an image. "
            "Text-only mode is limited. Please include an image in the query."
        )

    input_ids = hf_tokenizer(
        input_prompt, return_tensors="pt", padding=True, truncation=True
    ).to("cuda")
    inputs = {k: v.to(hf_model.device) for k, v in input_ids.items()}
    output = hf_model.generate(
        **input_ids, max_new_tokens=512, num_return_sequences=1, early_stopping=True
    )
    response_text = hf_tokenizer.decode(
        output[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
    )

    return response_text


async def ollama_model_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    kwargs.pop("max_tokens", None)
    kwargs.pop("response_format", None)

    ollama_client = ollama.AsyncClient()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await ollama_client.chat(model=model, messages=messages, **kwargs)

    result = response["message"]["content"]

    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": result, "model": model}})

    return result


async def gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def bedrock_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # Extract bedrock_model from kwargs if provided, otherwise use default
    bedrock_model = kwargs.pop('bedrock_model', "us.anthropic.claude-haiku-4-5-20251001-v1:0")
    return await bedrock_complete_if_cache(
        bedrock_model,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


def create_bedrock_complete(model: str):
    """
    Factory function to create a bedrock_complete function with a specific model.
    
    Args:
        model: Bedrock model ID (e.g., "us.anthropic.claude-haiku-4-5-20251001-v1:0")
    
    Returns:
        A configured bedrock_complete function
    """
    async def bedrock_complete_with_model(
        prompt, system_prompt=None, history_messages=[], **kwargs
    ) -> str:
        return await bedrock_complete_if_cache(
            model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )
    return bedrock_complete_with_model


@lru_cache(maxsize=1)
def initialize_hf_vision_pipeline(model_name):
    """
    Initialize an InternVL3 model for image-to-text inference.
    Returns a callable that processes (image, prompt) -> text.
    """
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    model_name_norm = _normalize_local_model_name(model_name)
    hf_model, hf_tokenizer = initialize_hf_model(model_name_norm)
    hf_model.to(device_str)
    hf_model.eval()
    log_gpu_memory("after InternVL3 load")

    def vision_inference(image, prompt: str = "Describe this image"):
        """Process image and prompt through InternVLChatModel."""
        try:            
            # InternVL3 expects pixel_values from images.
            # Use the model's internal image processing if available.
            if hasattr(hf_model, 'image_processor'):
                pixel_values = hf_model.image_processor(
                    image, return_tensors="pt"
                )["pixel_values"].to(device_str)
                logger.debug("pixel_values dtype=%s shape=%s", pixel_values.dtype, tuple(pixel_values.shape))
            else:
                # Prep the tensor using the custom function above
                # (Sets dynamic patching to 448x448 blocks, max 12 tiles)
                pixel_values = preprocess_internvl_image(image, input_size=448, max_num=12)

                # Move tensors to the same device/precision as your loaded hf_model
                pixel_values = pixel_values.to(device_str, dtype=torch.bfloat16)
            # else:
            #     # Fallback: assume image is PIL and use basic preprocessing
            #     from torchvision.transforms import ToTensor
            #     pixel_values = ToTensor()(image).unsqueeze(0).to(device_str).to(torch.bfloat16)
            #     print(f'DEBUG - 3.Type of pixel_values: {pixel_values.dtype}, shape: {pixel_values.shape}')
            
            # Generate response using the model's generate method.
            # Reset the peak counter so the snapshot below reports THIS forward's high-water
            # mark (the eager-attention score matrix is what spikes / OOMs).
            reset_gpu_peak()
            with torch.no_grad():
                generation_config = dict(max_new_tokens=1024, do_sample=True)
                response = hf_model.chat(
                    pixel_values=pixel_values,
                    question=prompt,
                    tokenizer=hf_tokenizer,
                    generation_config=generation_config,
                )
            log_gpu_memory(f"after InternVL3 generate ({pixel_values.shape[0]} tiles)")
            return response if isinstance(response, str) else str(response)
        except AttributeError:
            # Fallback if chat method is not available.
            raise RuntimeError(
                f"InternVLChatModel does not have expected chat interface for {model_name}"
            )
    
    return vision_inference



async def internvl3_14b_complete(
    prompt,
    system_prompt=None,
    history_messages=[],
    query_image_path: Optional[str] = None,
    **kwargs,
) -> str:
    model_name = kwargs.get("llm_model_name")
    hashing_kv = kwargs.get("hashing_kv")
    if not model_name and hashing_kv is not None and hasattr(hashing_kv, "global_config"):
        model_name = hashing_kv.global_config.get("llm_model_name")
    if not model_name:
        model_name = "./bin/pretrained/InternVL3-14B"

    if query_image_path:
        try:
            from .utils import resize_image_if_large

            usable_path = resize_image_if_large(query_image_path)
            image = Image.open(usable_path).convert("RGB")
            vision_infer = initialize_hf_vision_pipeline(model_name)
            kwargs.pop("hashing_kv", None)
            # vision_infer is now a callable that returns a string directly.
            output = vision_infer(image, prompt=prompt)
            return str(output).strip()
        except Exception as exc:
            logger.warning("InternVL3 image generation failed, falling back to text-only mode: %s", exc)

    if not model_name:
        raise ValueError(
            "internvl3_14b_complete could not resolve an HF model name or local path."
        )

    return await hf_model_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def ollama_model_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await ollama_model_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def openai_embedding(
    texts: list[str],
    model: str = "text-embedding-3-small",
    base_url: str = None,
    api_key: str = None,
) -> np.ndarray:
    # Use provided api_key, or fall back to environment variable
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    elif not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY not found. Please set it in environment variables or .env file, "
            "or pass it as api_key parameter."
        )

    openai_async_client = (
        AsyncOpenAI() if base_url is None else AsyncOpenAI(base_url=base_url)
    )
    response = await openai_async_client.embeddings.create(
        model=model, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


# @wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
# @retry(
#     stop=stop_after_attempt(3),
#     wait=wait_exponential(multiplier=1, min=4, max=10),
#     retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),  # TODO: fix exceptions
# )
async def bedrock_embedding(
    texts: list[str],
    model: str = "amazon.titan-embed-text-v2:0",
    aws_access_key_id=None,
    aws_secret_access_key=None,
    aws_session_token=None,
    aws_region=None,
) -> np.ndarray:
    # Only set environment variables if they are provided and not None
    # If not provided, boto3 will use default credentials (IAM role, ~/.aws/credentials, etc.)
    if aws_access_key_id is not None:
        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
    if aws_secret_access_key is not None:
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
    if aws_session_token is not None:
        os.environ["AWS_SESSION_TOKEN"] = aws_session_token
    
    # Get AWS region from parameter, environment variable, or use default
    region = aws_region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"

    session = aioboto3.Session()
    async with session.client("bedrock-runtime", region_name=region) as bedrock_async_client:
        if (model_provider := model.split(".")[0]) == "amazon":
            embed_texts = []
            for text in texts:
                if "v2" in model:
                    body = json.dumps(
                        {
                            "inputText": text,
                            # 'dimensions': embedding_dim,
                            "embeddingTypes": ["float"],
                        }
                    )
                elif "v1" in model:
                    body = json.dumps({"inputText": text})
                else:
                    raise ValueError(f"Model {model} is not supported!")

                response = await bedrock_async_client.invoke_model(
                    modelId=model,
                    body=body,
                    accept="application/json",
                    contentType="application/json",
                )

                response_body = await response.get("body").json()

                embed_texts.append(response_body["embedding"])
        elif model_provider == "cohere":
            body = json.dumps(
                {"texts": texts, "input_type": "search_document", "truncate": "NONE"}
            )

            response = await bedrock_async_client.invoke_model(
                model=model,
                body=body,
                accept="application/json",
                contentType="application/json",
            )

            response_body = json.loads(response.get("body").read())

            embed_texts = response_body["embeddings"]
        else:
            raise ValueError(f"Model provider '{model_provider}' is not supported!")

        return np.array(embed_texts)


def mean_pool_embeddings(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    attention_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
    summed = (token_embeddings * attention_mask).sum(dim=1)
    counts = attention_mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


async def hf_embedding(
    texts: list[str], tokenizer, embed_model, pooling: str = "cls"
) -> np.ndarray:
    """Encode texts with a HuggingFace model and L2-normalize the result.

    pooling:
      - "cls" (default): run the full transformer and take the [CLS] token of
        the last hidden state. This is the *contextual* sentence embedding and
        the pooling BGE/BGE-M3 are trained with -- use it for any real encoder.
      - "mean": run the transformer and mean-pool the last hidden state over the
        attention mask.
      - "input_mean": mean-pool the *static* input-embedding lookup table without
        running the transformer. This is a degenerate bag-of-token-vectors with
        no contextualization; only meaningful as a fallback for decoder/VLM
        models (e.g. InternVL) that are not encoders.

    NOTE: ``input_mean`` was previously selected automatically via
    ``hasattr(embed_model, "get_input_embeddings")``, but *every* HF model has
    that attribute, so BGE-M3 silently fell into the static-lookup path -- the
    transformer was never run and retrieval was essentially random. Pooling is
    now explicit so the stored index and queries always use the same encoder.
    """
    device = next(embed_model.parameters()).device if hasattr(embed_model, "parameters") else torch.device("cpu")
    encoded = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    with torch.no_grad():
        if pooling == "input_mean":
            token_embeddings = embed_model.get_input_embeddings()(encoded["input_ids"])
            embeddings = mean_pool_embeddings(token_embeddings, encoded["attention_mask"])
        else:
            outputs = embed_model(**encoded)
            if pooling == "mean":
                embeddings = mean_pool_embeddings(
                    outputs.last_hidden_state, encoded["attention_mask"]
                )
            elif pooling == "cls":
                embeddings = outputs.last_hidden_state[:, 0]
            else:
                raise ValueError(f"Unknown pooling strategy: {pooling!r}")
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    return embeddings.detach().cpu().to(torch.float32).numpy()


def create_bge_m3_embedding(
    model_name: str = "./bin/pretrained/bge-m3",
    device: str = "cpu",
):
    model_name_norm = _normalize_local_model_name(model_name)

    @wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
    async def bge_m3_embedding(texts: list[str]) -> np.ndarray:
        embed_model, tokenizer = initialize_hf_text_encoder(model_name_norm, device=device)
        # BGE-M3 is trained with CLS pooling + L2 normalization; anything else
        # (especially the old static-lookup path) wrecks retrieval quality.
        return await hf_embedding(texts, tokenizer, embed_model, pooling="cls")

    return bge_m3_embedding


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    """Standard normalization and resizing transform for InternVL"""
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Finds the best layout grid for dynamic high-res tile slicing"""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def preprocess_internvl_image(image, input_size=448, max_num=12):
    """
    Loads an image, dynamically crops it into multiple 448x448 tiles,
    and appends a global thumbnail tile.
    """
    transform = build_transform(input_size=input_size)
    
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Generate candidate grid layouts (e.g., 1x2, 2x3 tiles) up to max_num blocks
    target_ratios = set()
    for i in range(1, max_num + 1):
        for j in range(1, max_num + 1):
            if i * j <= max_num:
                target_ratios.add((i, j))
    target_ratios = sorted(list(target_ratios), key=lambda x: x[0] * x[1])

    # Find best grid layout for current aspect ratio
    target_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, input_size
    )

    # Calculate dimensions for the cropped patches
    target_width = input_size * target_ratio[0]
    target_height = input_size * target_ratio[1]
    blocks = target_ratio[0] * target_ratio[1]

    # Resize image to fit the new layout grid
    resized_img = image.resize((target_width, target_height), Image.BILINEAR)
    processed_images = []
    
    # Slice into sub-tiles
    for i in range(blocks):
        box = (
            (i % target_ratio[0]) * input_size,
            (i // target_ratio[0]) * input_size,
            ((i % target_ratio[0]) + 1) * input_size,
            ((i // target_ratio[0]) + 1) * input_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(transform(split_img))

    # Always append a global thumbnail (resized version of the whole image)
    thumbnail_img = image.resize((input_size, input_size), Image.BILINEAR)
    processed_images.append(transform(thumbnail_img))

    # Stack into a single tensor of shape: (num_tiles, 3, 448, 448)
    pixel_values = torch.stack(processed_images)
    return pixel_values


if __name__ == "__main__":
    import asyncio

    async def main():
        result = await gpt_4o_mini_complete("How are you?")
        print(result)

    asyncio.run(main())
