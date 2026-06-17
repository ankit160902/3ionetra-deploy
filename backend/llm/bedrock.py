"""
AWS Bedrock (Converse API) helper — single integration point for all LLM calls.

Replaces the previous Google Gemini (google-genai) integration. Every runtime
call site routes through ``bedrock_generate`` / ``bedrock_stream`` so the rest of
the codebase stays provider-agnostic.

Auth: standard AWS credential chain (IAM role on ECS/Fargate, or ~/.aws locally).
No API key — ensure the task/role has ``bedrock:InvokeModel*`` and that model
access is enabled for the chosen model ids in the Bedrock console.

Return shape mimics the minimal subset of the old Gemini response object that
callers used (``.text`` and ``.usage_metadata.prompt_token_count`` /
``.candidates_token_count``) so call sites needed only a one-line swap.
"""

import logging
from functools import lru_cache

import boto3

from config import settings

logger = logging.getLogger(__name__)

# mime-type → Bedrock image "format" token
_IMAGE_FORMAT_BY_MIME = {
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
}


@lru_cache(maxsize=1)
def get_bedrock_runtime():
    """Cached boto3 bedrock-runtime client."""
    return boto3.client("bedrock-runtime", region_name=settings.BEDROCK_REGION)


class _Usage:
    """Mimics google-genai usage_metadata for the two fields callers read."""
    __slots__ = ("prompt_token_count", "candidates_token_count")

    def __init__(self, input_tokens: int, output_tokens: int):
        self.prompt_token_count = input_tokens
        self.candidates_token_count = output_tokens


class BedrockResult:
    """Minimal stand-in for the old Gemini response object (``.text``)."""
    __slots__ = ("text", "usage_metadata")

    def __init__(self, text: str, usage_metadata: "_Usage | None" = None):
        self.text = text
        self.usage_metadata = usage_metadata


def _content_blocks(prompt, images=None, video=None):
    """Build a Converse message content list: media blocks first, then text."""
    blocks = []
    if images:
        for data, mime in images:
            fmt = _IMAGE_FORMAT_BY_MIME.get((mime or "").lower(), "jpeg")
            blocks.append({"image": {"format": fmt, "source": {"bytes": data}}})
    if video:
        data, fmt = video
        blocks.append({"video": {"format": fmt, "source": {"bytes": data}}})
    blocks.append({"text": prompt})
    return blocks


def _request_kwargs(prompt, *, model, system, temperature, max_tokens, images=None, video=None):
    kwargs = {
        "modelId": model,
        "messages": [{"role": "user", "content": _content_blocks(prompt, images, video)}],
        "inferenceConfig": {"temperature": float(temperature), "maxTokens": int(max_tokens)},
    }
    if system:
        kwargs["system"] = [{"text": system}]
    return kwargs


def bedrock_generate(
    prompt: str,
    *,
    model: str,
    system: "str | None" = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    json_mode: bool = False,
    images=None,
    video=None,
) -> BedrockResult:
    """Single-shot Converse call. Returns a BedrockResult with ``.text``.

    ``json_mode`` nudges the model toward raw JSON (Claude has no native JSON
    mode — callers parse with the existing ``extract_json`` helper).
    """
    if json_mode:
        prompt = prompt + "\n\nReturn ONLY a valid JSON object. No prose, no markdown fences."
    client = get_bedrock_runtime()
    resp = client.converse(
        **_request_kwargs(
            prompt, model=model, system=system,
            temperature=temperature, max_tokens=max_tokens,
            images=images, video=video,
        )
    )
    parts = resp.get("output", {}).get("message", {}).get("content", [])
    text = "".join(p.get("text", "") for p in parts)
    usage = resp.get("usage", {}) or {}
    return BedrockResult(text, _Usage(usage.get("inputTokens", 0), usage.get("outputTokens", 0)))


def bedrock_stream(
    prompt: str,
    *,
    model: str,
    system: "str | None" = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
):
    """Streaming Converse call — sync generator yielding text deltas (str)."""
    client = get_bedrock_runtime()
    resp = client.converse_stream(
        **_request_kwargs(
            prompt, model=model, system=system,
            temperature=temperature, max_tokens=max_tokens,
        )
    )
    for event in resp.get("stream", []):
        delta = event.get("contentBlockDelta", {}).get("delta", {})
        if "text" in delta:
            yield delta["text"]
