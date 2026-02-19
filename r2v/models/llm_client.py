"""
Unified LLM client supporting multiple providers for teacher trajectory collection.

Supported providers:
- OpenAI (GPT-4o, GPT-4-turbo, etc.)
- Anthropic (Claude 3.5 Sonnet, Claude 3 Opus, etc.)
- Google Gemini (gemini-1.5-pro, gemini-2.0-flash, etc.)
- DeepSeek (deepseek-chat, deepseek-reasoner, etc.)

All API keys are read from a .env file or environment variables.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential

logger = logging.getLogger(__name__)

# Load .env from project root (walks up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


# ── Data structures ──────────────────────────────────────────────


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""

    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    finish_reason: str = ""
    raw: Any = None  # Provider-specific raw response


@dataclass
class LLMConfig:
    """Configuration for an LLM provider."""

    provider: str  # "openai", "anthropic", "google", "deepseek"
    model_name: str
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 4096
    system_prompt: str = ""
    extra_params: dict[str, Any] = field(default_factory=dict)


# ── Cost estimation (approximate $/1M tokens) ────────────────────

_COST_PER_1M = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    # Anthropic
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    # Google
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    # DeepSeek
    "deepseek-chat": {"input": 0.14, "output": 0.28},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate API cost in USD."""
    costs = _COST_PER_1M.get(model, {"input": 1.0, "output": 3.0})
    return (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1_000_000


# ── Abstract base ────────────────────────────────────────────────


class BaseLLMClient(ABC):
    """Abstract LLM client interface."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a completion for the given prompt."""
        ...

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        """Generate a chat completion for the given messages."""
        ...

    def _merge_kwargs(self, **kwargs) -> dict:
        """Merge call-time kwargs with config defaults."""
        return {
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            **self.config.extra_params,
            **{k: v for k, v in kwargs.items() if k not in ("temperature", "top_p", "max_tokens")},
        }


# ── OpenAI ───────────────────────────────────────────────────────


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI API (GPT-4o, GPT-4-turbo, etc.)."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        import openai

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment. Add it to your .env file.")
        self.client = openai.OpenAI(api_key=api_key)

    @retry(wait=wait_random_exponential(min=10, max=120), stop=stop_after_attempt(6))
    def chat(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        params = self._merge_kwargs(**kwargs)
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=params["temperature"],
            top_p=params["top_p"],
            max_tokens=params["max_tokens"],
        )
        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            text=choice.message.content or "",
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cost=estimate_cost(
                self.config.model_name,
                usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0,
            ),
            finish_reason=choice.finish_reason or "",
            raw=response,
        )

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kwargs)


# ── Anthropic ────────────────────────────────────────────────────


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic API (Claude 3.5 Sonnet, Claude 3 Opus, etc.)."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment. Add it to your .env file.")
        self.client = anthropic.Anthropic(api_key=api_key)

    @retry(wait=wait_random_exponential(min=10, max=120), stop=stop_after_attempt(6))
    def chat(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        params = self._merge_kwargs(**kwargs)

        # Separate system message from user/assistant messages
        system = self.config.system_prompt
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append(msg)

        create_kwargs: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": chat_messages,
            "max_tokens": params["max_tokens"],
            "temperature": params["temperature"],
            "top_p": params["top_p"],
        }
        if system:
            create_kwargs["system"] = system

        response = self.client.messages.create(**create_kwargs)
        text = response.content[0].text if response.content else ""
        return LLMResponse(
            text=text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost=estimate_cost(
                self.config.model_name,
                response.usage.input_tokens,
                response.usage.output_tokens,
            ),
            finish_reason=response.stop_reason or "",
            raw=response,
        )

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)


# ── Google Gemini ────────────────────────────────────────────────


class GeminiClient(BaseLLMClient):
    """Client for Google Gemini API."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment. Add it to your .env file.")

        import google.generativeai as genai

        genai.configure(api_key=api_key)
        generation_config = genai.types.GenerationConfig(
            temperature=config.temperature,
            top_p=config.top_p,
            max_output_tokens=config.max_tokens,
        )
        system_instruction = config.system_prompt if config.system_prompt else None
        self.model = genai.GenerativeModel(
            model_name=config.model_name,
            generation_config=generation_config,
            system_instruction=system_instruction,
        )
        self._genai = genai

    @retry(wait=wait_random_exponential(min=10, max=120), stop=stop_after_attempt(6))
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        response = self.model.generate_content(prompt)
        # Extract usage metadata
        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
        output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
        return LLMResponse(
            text=response.text if response.text else "",
            model=self.config.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=estimate_cost(self.config.model_name, input_tokens, output_tokens),
            finish_reason=str(getattr(response.candidates[0], "finish_reason", "")) if response.candidates else "",
            raw=response,
        )

    @retry(wait=wait_random_exponential(min=10, max=120), stop=stop_after_attempt(6))
    def chat(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        # Convert OpenAI-style messages to Gemini format
        gemini_messages = []
        for msg in messages:
            role = msg["role"]
            if role == "system":
                # System messages are handled via model config; prepend to first user msg
                continue
            gemini_role = "user" if role == "user" else "model"
            gemini_messages.append({"role": gemini_role, "parts": [msg["content"]]})

        chat = self.model.start_chat(history=gemini_messages[:-1])
        last_msg = gemini_messages[-1]["parts"][0] if gemini_messages else ""
        response = chat.send_message(last_msg)

        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
        output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
        return LLMResponse(
            text=response.text if response.text else "",
            model=self.config.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=estimate_cost(self.config.model_name, input_tokens, output_tokens),
            finish_reason=str(getattr(response.candidates[0], "finish_reason", "")) if response.candidates else "",
            raw=response,
        )


# ── DeepSeek ─────────────────────────────────────────────────────


class DeepSeekClient(BaseLLMClient):
    """Client for DeepSeek API (OpenAI-compatible endpoint)."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        import openai

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment. Add it to your .env file.")
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )

    @retry(wait=wait_random_exponential(min=10, max=120), stop=stop_after_attempt(6))
    def chat(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        params = self._merge_kwargs(**kwargs)
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=params["temperature"],
            top_p=params["top_p"],
            max_tokens=params["max_tokens"],
        )
        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            text=choice.message.content or "",
            model=response.model or self.config.model_name,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cost=estimate_cost(
                self.config.model_name,
                usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0,
            ),
            finish_reason=choice.finish_reason or "",
            raw=response,
        )

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kwargs)


# ── Factory ──────────────────────────────────────────────────────

_CLIENT_REGISTRY: dict[str, type[BaseLLMClient]] = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
    "google": GeminiClient,
    "gemini": GeminiClient,
    "deepseek": DeepSeekClient,
}


def create_llm_client(config: LLMConfig) -> BaseLLMClient:
    """Create an LLM client from a config."""
    provider = config.provider.lower()
    if provider not in _CLIENT_REGISTRY:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported: {list(_CLIENT_REGISTRY.keys())}"
        )
    return _CLIENT_REGISTRY[provider](config)


def create_llm_client_from_cfg(cfg) -> BaseLLMClient:
    """Create an LLM client from an OmegaConf teacher config section.

    Expected config structure:
        teacher:
            provider: openai        # or anthropic, google, deepseek
            model_name: gpt-4o
            temperature: 0.7
            top_p: 0.95
            max_tokens: 4096
            system_prompt: "..."     # optional
    """
    teacher_cfg = cfg.get("teacher", cfg)
    llm_config = LLMConfig(
        provider=teacher_cfg.get("provider", _infer_provider(teacher_cfg.get("model_name", ""))),
        model_name=teacher_cfg.get("model_name", "gpt-4o"),
        temperature=teacher_cfg.get("temperature", 0.7),
        top_p=teacher_cfg.get("top_p", 0.95),
        max_tokens=teacher_cfg.get("max_tokens", 4096),
        system_prompt=teacher_cfg.get("system_prompt", ""),
    )
    logger.info(f"Creating LLM client: provider={llm_config.provider}, model={llm_config.model_name}")
    return create_llm_client(llm_config)


def _infer_provider(model_name: str) -> str:
    """Infer provider from model name if not explicitly given."""
    model_lower = model_name.lower()
    if any(x in model_lower for x in ("gpt", "o1", "o3")):
        return "openai"
    if any(x in model_lower for x in ("claude",)):
        return "anthropic"
    if any(x in model_lower for x in ("gemini",)):
        return "google"
    if any(x in model_lower for x in ("deepseek",)):
        return "deepseek"
    # Default to OpenAI-compatible
    return "openai"
