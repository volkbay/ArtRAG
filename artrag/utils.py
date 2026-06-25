import asyncio
import base64
import html
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from functools import wraps
from hashlib import md5
from pathlib import Path
from typing import Any, Union
import xml.etree.ElementTree as ET

import numpy as np
import tiktoken
from PIL import Image

# Source art scans can be enormous (artpedia images reach hundreds of MB and
# tens of thousands of pixels). Disable Pillow's decompression-bomb guard so we
# can open them; we always downscale explicitly via resize_image_if_large below.
Image.MAX_IMAGE_PIXELS = None

ENCODER = None

logger = logging.getLogger("lightrag")


_DETAIL_FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def set_logger(log_file: str):
    """Three-tier logging.

    * console handler (INFO, compact)         -> concise per-sample flow on the terminal
    * persistent `log_file` handler (INFO)    -> general cross-sample flow (lightrag.log)
    * per-sample handlers (DEBUG, attached on
      demand via add_sample_logfile)          -> full detail for one sample

    The full DEBUG trail (descriptions, scores, VLM/final ranking, summaries) is
    captured by the per-sample handler, NOT lightrag.log, so the flow log stays
    small. The terminal never shows DEBUG. Re-running is idempotent.
    """
    logger.setLevel(logging.DEBUG)
    # Don't propagate to the root logger: a dependency may call logging.basicConfig
    # (adding a root handler), which would otherwise duplicate every line AND leak
    # our DEBUG detail onto the console. Our own handlers are the only sinks.
    logger.propagate = False

    if logger.handlers:
        return

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(_DETAIL_FORMATTER)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)


def add_sample_logfile(path: str) -> logging.Handler:
    """Attach a per-sample DEBUG file handler capturing the full detail trail.

    Always detailed regardless of the `verbose` setting (verbose only governs
    terminal transformer noise). Pair every call with remove_sample_logfile.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    handler = logging.FileHandler(path, mode="w")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(_DETAIL_FORMATTER)
    logger.addHandler(handler)
    return handler


def remove_sample_logfile(handler: logging.Handler) -> None:
    """Detach and close a handler returned by add_sample_logfile."""
    if handler is None:
        return
    handler.flush()
    logger.removeHandler(handler)
    handler.close()

def log_gpu_memory(tag: str) -> None:
    """Log a one-line VRAM snapshot at `tag`, gated by settings.log_gpu_memory.

    Reports both PyTorch's view and the device's view so the gap is visible:
      * alloc   = tensors currently live in PyTorch
      * peak    = max alloc since the last reset_peak_memory_stats (the forward's high-water
                  mark — this is what actually OOMs, not the resting `alloc`)
      * reserved= PyTorch's cache (alloc + reserved-but-free)
      * used/total = whole-device (nvidia-smi view); used-reserved ≈ other procs/driver.
    """
    from .runtime_config import settings

    if not getattr(settings, "log_gpu_memory", False):
        return
    try:
        import torch

        if not torch.cuda.is_available():
            return
        dev = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(dev)
        gb = 1024 ** 3
        logger.info(
            "VRAM [%s] alloc=%.2f peak=%.2f reserved=%.2f | device used=%.2f/%.2f free=%.2f GiB",
            tag,
            torch.cuda.memory_allocated(dev) / gb,
            torch.cuda.max_memory_allocated(dev) / gb,
            torch.cuda.memory_reserved(dev) / gb,
            (total - free) / gb,
            total / gb,
            free / gb,
        )
    except Exception as e:  # diagnostics must never break the run
        logger.debug("VRAM probe failed at %s: %s", tag, e)


def reset_gpu_peak() -> None:
    """Reset PyTorch's peak-allocation counter so the next log_gpu_memory `peak` reflects
    only the work since this call (e.g. one rerank forward). No-op unless logging is on."""
    from .runtime_config import settings

    if not getattr(settings, "log_gpu_memory", False):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute a unique ID for a given content string.

    The ID is a combination of the given prefix and the MD5 hash of the content string.
    """
    return prefix + md5(content.encode()).hexdigest()

@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)


def locate_json_string_body_from_string(content: str) -> Union[str, None]:
    """Locate the JSON string body from a string"""
    maybe_json_str = re.search(r"{.*}", content, re.DOTALL)
    if maybe_json_str is not None:
        return maybe_json_str.group(0)
    else:
        return None


def convert_response_to_json(response: str) -> dict:
    json_str = locate_json_string_body_from_string(response)
    assert json_str is not None, f"Unable to parse JSON from response: {response}"
    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {json_str}")
        raise e from None


def compute_args_hash(*args):
    return md5(str(args).encode()).hexdigest()


def limit_async_func_call(max_size: int, waitting_time: float = 0.0001):
    """Add restriction of maximum async calling times for a async func"""

    def final_decro(func):
        """Not using async.Semaphore to aovid use nest-asyncio"""
        __current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            while __current_size >= max_size:
                await asyncio.sleep(waitting_time)
            __current_size += 1
            result = await func(*args, **kwargs)
            __current_size -= 1
            return result

        return wait_func

    return final_decro


def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro


def load_json(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


def write_json(json_obj, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    tokens = ENCODER.encode(content)
    return tokens


def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    content = ENCODER.decode(tokens)
    return content


def pack_user_ass_to_openai_messages(*args: str):
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


# Refer the utils functions of the official GraphRAG implementation:
# https://github.com/microsoft/graphrag
def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


def is_float_regex(value):
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data


def save_data_to_file(data, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def xml_to_json(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Print the root element's tag and attributes to confirm the file has been correctly loaded
        print(f"Root element: {root.tag}")
        print(f"Root attributes: {root.attrib}")

        data = {
            "nodes": [],
            "edges": []
        }

        # Use namespace
        namespace = {'': 'http://graphml.graphdrawing.org/xmlns'}

        for node in root.findall('.//node', namespace):
            node_data = {
                "id": node.get('id').strip('"'),
                "entity_type": node.find("./data[@key='d0']", namespace).text.strip('"') if node.find("./data[@key='d0']", namespace) is not None else "",
                "description": node.find("./data[@key='d1']", namespace).text if node.find("./data[@key='d1']", namespace) is not None else "",
                "source_id": node.find("./data[@key='d2']", namespace).text if node.find("./data[@key='d2']", namespace) is not None else ""
            }
            data["nodes"].append(node_data)

        for edge in root.findall('.//edge', namespace):
            edge_data = {
                "source": edge.get('source').strip('"'),
                "target": edge.get('target').strip('"'),
                "weight": float(edge.find("./data[@key='d3']", namespace).text) if edge.find("./data[@key='d3']", namespace) is not None else 0.0,
                "description": edge.find("./data[@key='d4']", namespace).text if edge.find("./data[@key='d4']", namespace) is not None else "",
                "keywords": edge.find("./data[@key='d5']", namespace).text if edge.find("./data[@key='d5']", namespace) is not None else "",
                "source_id": edge.find("./data[@key='d6']", namespace).text if edge.find("./data[@key='d6']", namespace) is not None else ""
            }
            data["edges"].append(edge_data)

        # Print the number of nodes and edges found
        print(f"Found {len(data['nodes'])} nodes and {len(data['edges'])} edges")

        return data
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    



def list_of_list_to_csv(data: list[list]):
    return "\n".join(
        [",\t".join([str(data_dd) for data_dd in data_d]) for data_d in data]
    )


def generate_context_sections(node_datas, use_relations):
    """
    Generates CSV context sections for entities and relationships.
    """
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    relations_section_list = [
        ["id", "source", "target", "description", "weight"]
    ]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["weight"],
                # e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    return entities_context, relations_context


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode image file to base64 string

    Args:
        image_path: Path to the image file

    Returns:
        str: Base64 encoded string, empty string if encoding fails
    """
    try:
        usable_path = resize_image_if_large(image_path)
        with open(usable_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        return ""


def validate_image_file(image_path: str, max_size_mb: int = 50) -> bool:
    """
    Validate if a file is a valid image file

    Args:
        image_path: Path to the image file
        max_size_mb: Maximum file size in MB

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        path = Path(image_path)

        # Check if file exists
        if not path.exists():
            logger.warning(f"Image file not found: {image_path}")
            return False

        # Check file extension
        image_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".webp",
            ".tiff",
            ".tif",
        ]

        path_lower = str(path).lower()
        has_valid_extension = any(path_lower.endswith(ext) for ext in image_extensions)

        if not has_valid_extension:
            logger.warning(f"File does not appear to be an image: {image_path}")
            return False

        # Check file size
        file_size = path.stat().st_size
        max_size = max_size_mb * 1024 * 1024

        if file_size > max_size:
            logger.warning(f"Image file too large ({file_size} bytes): {image_path}")
            return False

        # Only log successful validation at DEBUG level (not shown with INFO level)
        return True

    except Exception as e:
        logger.error(f"Error validating image file {image_path}: {e}")
        return False


def resize_image_if_large(
    image_path: str, max_dimension: int = 2048, output_dir: str = None
) -> str:
    """
    Downscale an image whose longest side exceeds ``max_dimension``.

    Very large source scans waste memory, slow inference, and can exceed API
    upload limits. When the longest side is over ``max_dimension`` the image is
    written to a resized JPEG copy (aspect ratio preserved) and that copy's path
    is returned; otherwise the original path is returned unchanged. The original
    file is never modified.

    Args:
        image_path: Path to the source image.
        max_dimension: Maximum allowed length (px) of the longest side.
        output_dir: Directory for the resized copy. Defaults to the system temp
            directory.

    Returns:
        str: Path to an image whose longest side is <= ``max_dimension``. Falls
        back to the original path if resizing fails.
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            longest = max(width, height)
            if longest <= max_dimension:
                return str(image_path)

            scale = max_dimension / longest
            new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            resized = img.convert("RGB").resize(new_size, Image.LANCZOS)

        out_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir())
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{Path(image_path).stem}_resized_{new_size[0]}x{new_size[1]}.jpg"
        resized.save(out_path, format="JPEG", quality=95)
        logger.info(
            f"Resized large image {image_path} ({width}x{height}) -> "
            f"{out_path} ({new_size[0]}x{new_size[1]})"
        )
        return str(out_path)
    except Exception as e:
        logger.error(f"Failed to resize image {image_path}: {e}")
        return str(image_path)
