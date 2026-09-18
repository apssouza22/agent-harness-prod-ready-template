"""Compatibility patches for mem0 AWS Bedrock OpenAI foundation models."""

from __future__ import annotations

from typing import Any

from src.app.core.llm.factory import build_bedrock_converse_inference_config

_PATCHED = False


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "items"):
        return dict(value)
    return {key: getattr(value, key) for key in ("text", "reasoningContent", "reasoning_content") if hasattr(value, key)}


def _get_converse_content_blocks(response: Any) -> list[dict[str, Any]]:
    message: Any = None
    if hasattr(response, "output") and hasattr(response.output, "message"):
        message = response.output.message
    elif isinstance(response, dict):
        message = response.get("output", {}).get("message")

    if message is None:
        return []

    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", [])
    return [_coerce_mapping(block) for block in content or []]


def _extract_direct_text(block: dict[str, Any]) -> str | None:
    text_value = block.get("text")
    if isinstance(text_value, str):
        return text_value
    if isinstance(text_value, dict):
        nested_text = text_value.get("text")
        if isinstance(nested_text, str):
            return nested_text
    return None


def _extract_reasoning_text(block: dict[str, Any]) -> str | None:
    reasoning = block.get("reasoningContent") or block.get("reasoning_content")
    if not isinstance(reasoning, dict):
        return None

    reasoning_text = reasoning.get("reasoningText") or reasoning.get("reasoning_text")
    if isinstance(reasoning_text, dict):
        text = reasoning_text.get("text")
        if isinstance(text, str):
            return text
    return None


def _extract_converse_text(response: Any) -> str:
    answer_parts: list[str] = []
    reasoning_parts: list[str] = []

    for block in _get_converse_content_blocks(response):
        direct_text = _extract_direct_text(block)
        if direct_text:
            answer_parts.append(direct_text)
            continue

        reasoning_text = _extract_reasoning_text(block)
        if reasoning_text:
            reasoning_parts.append(reasoning_text)

    if answer_parts:
        return answer_parts[-1]
    if reasoning_parts:
        return reasoning_parts[-1]

    raise ValueError("bedrock_converse_response_missing_text_content")


def _format_converse_messages(messages: list[dict[str, str]]) -> tuple[list[dict[str, Any]], str | None]:
    formatted_messages: list[dict[str, Any]] = []
    system_message: str | None = None

    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            system_message = content
        elif role in {"user", "assistant"}:
            formatted_messages.append({"role": role, "content": [{"text": content}]})

    return formatted_messages, system_message


def apply_mem0_bedrock_openai_compat() -> None:
    """Patch mem0 so Bedrock OpenAI model ids initialize and use Converse."""
    global _PATCHED
    if _PATCHED:
        return

    import mem0.llms.aws_bedrock as bedrock_module

    if "openai" not in bedrock_module.PROVIDERS:
        bedrock_module.PROVIDERS.append("openai")

    original_generate_standard = bedrock_module.AWSBedrockLLM._generate_standard

    def _generate_standard_with_openai_support(self, messages, stream=False):
        if self.provider != "openai":
            return original_generate_standard(self, messages, stream)

        formatted_messages, system_message = _format_converse_messages(messages)
        inference_config = build_bedrock_converse_inference_config(
            self.config.model,
            max_tokens=self.model_config.get("max_tokens", 2000),
            temperature=self.model_config.get("temperature", 0.1),
            top_p=self.model_config.get("top_p", 0.9),
        )

        converse_params: dict[str, Any] = {
            "modelId": self.config.model,
            "messages": formatted_messages,
            "inferenceConfig": inference_config,
        }
        if system_message:
            converse_params["system"] = [{"text": system_message}]

        response = self.client.converse(**converse_params)
        return _extract_converse_text(response)

    bedrock_module.AWSBedrockLLM._generate_standard = _generate_standard_with_openai_support
    _PATCHED = True
