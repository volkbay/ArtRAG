import os
import copy
import json
import aioboto3
import numpy as np
from typing import List, Optional
from pathlib import Path
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import torch
from .base import BaseKVStorage
from .utils import compute_args_hash, wrap_embedding_func_with_attrs
import base64
import pdb
from functools import lru_cache

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
    AsyncOpenAI,
    APIConnectionError,
    RateLimitError,
    Timeout,
    APITimeoutError,
    AsyncAzureOpenAI,
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
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
    hf_tokenizer = AutoTokenizer.from_pretrained(
        model_name, device_map="auto"
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto"
    )
    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token

    return hf_model, hf_tokenizer


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
    model_name = model
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


async def hf_model_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
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


async def hf_embedding(texts: list[str], tokenizer, embed_model) -> np.ndarray:
    input_ids = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    ).input_ids
    with torch.no_grad():
        outputs = embed_model(input_ids)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()


async def ollama_embedding(texts: list[str], embed_model, **kwargs) -> np.ndarray:
    """
    Deprecated in favor of `embed`.
    """
    embed_text = []
    ollama_client = ollama.Client(**kwargs)
    for text in texts:
        data = ollama_client.embeddings(model=embed_model, prompt=text)
        embed_text.append(data["embedding"])

    return embed_text


if __name__ == "__main__":
    import asyncio

    async def main():
        result = await gpt_4o_mini_complete("How are you?")
        print(result)

    asyncio.run(main())
