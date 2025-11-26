"""
Base Classes and Core Definitions for LLM Inference

This module provides the foundational classes and data structures for standardized
LLM inference across multiple providers. It defines the input/output contracts
and base functionality that all LLM clients should implement.

The design follows a provider-agnostic approach with clear separation between
configuration, request formatting, and response handling.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum
import json


class LLMProvider(str, Enum):
    """
    Enumeration of supported LLM service providers.

    This ensures type-safe provider selection and consistent naming across
    the codebase.
    """

    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    DOUBAO = "doubao"
    QWEN = "qwen"
    AIHUBMIX = "aihubmix"
    CUSTOM = "custom"


@dataclass
class Message:
    """
    Represents a single message in a conversation with an LLM.

    Follows the standard chat message format with role-based content
    organization.

    Attributes:
        role (str): The role of the message sender. Common values include:
                   'system', 'user', 'assistant', 'developer'
        content (str): The actual text content of the message
        name (Optional[str]): Optional name identifier for the speaker
    """

    role: str
    content: str
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert Message to dictionary format for API consumption."""
        result = {"role": self.role, "content": self.content}
        if self.name:
            result["name"] = self.name
        return result


@dataclass
class LLMRequest:
    """
    Standardized request structure for LLM inference calls.

    This dataclass encapsulates all possible parameters for LLM generation
    in a provider-agnostic way. Default values are set for common use cases.

    Attributes:
        model (str): The model identifier for the LLM provider
        messages (List[Message]): Conversation history and current prompt
        temperature (float): Controls randomness (0.0-1.0). Lower = more deterministic
        max_tokens (int): Maximum number of tokens to generate
        top_p (float): Nucleus sampling parameter (0.0-1.0)
        frequency_penalty (float): Penalize frequent tokens (-2.0 to 2.0)
        presence_penalty (float): Penalize new token presence (-2.0 to 2.0)
        stop (Optional[List[str]]): Sequences where generation should stop
        seed (Optional[int]): Random seed for reproducible results
        stream (bool): Whether to stream the response
        extra_params (Dict[str, Any]): Provider-specific additional parameters
    """

    model: str
    messages: List[Message] = field(default_factory=list)
    temperature: float = 0.1
    max_tokens: int = 1024
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    seed: Optional[int] = None
    stream: bool = False
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_api_format(self) -> Dict[str, Any]:
        """
        Convert the request to provider-agnostic API format.

        Returns:
            Dictionary ready for serialization to JSON for API calls
        """
        base_params = {
            "model": self.model,
            "messages": [msg.to_dict() for msg in self.messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stream": self.stream,
        }

        # Add optional parameters if they are provided
        if self.stop is not None:
            base_params["stop"] = self.stop
        if self.seed is not None:
            base_params["seed"] = self.seed

        # Merge with any extra parameters
        base_params.update(self.extra_params)

        return base_params

    @classmethod
    def from_simple_prompt(cls, model: str, prompt: str, **kwargs) -> "LLMRequest":
        """
        Create an LLMRequest from a simple text prompt.

        Args:
            model: The model identifier to use
            prompt: The user prompt text
            **kwargs: Additional parameters for LLMRequest

        Returns:
            LLMRequest configured with the prompt as a user message
        """
        message = Message(role="user", content=prompt)
        return cls(model=model, messages=[message], **kwargs)


@dataclass
class LLMResponse:
    """
    Standardized response structure from LLM inference calls.

    This dataclass provides a consistent interface for handling LLM responses
    regardless of the underlying provider.

    Attributes:
        content (str): The generated text content
        model (str): The model that generated the response
        usage (Dict[str, int]): Token usage statistics
        finish_reason (str): Reason why generation stopped
        raw_response (Any): The original provider response object
    """

    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"
    raw_response: Any = None

    @property
    def prompt_tokens(self) -> int:
        """Get the number of tokens in the prompt."""
        return self.usage.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        """Get the number of tokens in the completion."""
        return self.usage.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        """Get the total number of tokens used."""
        return self.usage.get("total_tokens", 0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for easy serialization."""
        return {
            "content": self.content,
            "model": self.model,
            "usage": self.usage,
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"LLMResponse(model={self.model}, content={self.content[:100]}..., tokens={self.total_tokens})"


class BaseLLMClient:
    """
    Abstract base class for all LLM clients.

    This class defines the common interface that all provider-specific
    LLM clients should implement. It ensures consistency across different
    providers and enables easy switching between them.
    """

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        """
        Initialize the base LLM client.

        Args:
            api_key: Authentication key for the LLM service
            base_url: Optional base URL for the API endpoint
        """
        self.api_key = api_key
        self.base_url = base_url
        self.client = None

    def initialize_client(self) -> Any:
        """
        Initialize the provider-specific client.

        Returns:
            Initialized client object specific to the provider

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement initialize_client")

    def call(self, request: LLMRequest) -> LLMResponse:
        """
        Execute an LLM inference call.

        Args:
            request: Standardized LLM request parameters

        Returns:
            Standardized LLM response

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement call")

    def validate_request(self, request: LLMRequest) -> None:
        """
        Validate the LLM request before sending.

        Args:
            request: The LLM request to validate

        Raises:
            ValueError: If the request is invalid
        """
        if not request.model:
            raise ValueError("Model must be specified")
        if not request.messages:
            raise ValueError("At least one message must be provided")
        if request.temperature < 0 or request.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        if request.max_tokens <= 0:
            raise ValueError("Max tokens must be positive")


def detect_provider(model_name: str) -> LLMProvider:
    """
    Detect the LLM provider based on model name patterns.

    Args:
        model_name: The model identifier string

    Returns:
        Detected LLMProvider enum value
    """
    model_lower = model_name.lower()

    if "doubao" in model_lower:
        return LLMProvider.DOUBAO
    elif "deepseek" in model_lower:
        return LLMProvider.DEEPSEEK
    elif "gpt" in model_lower:
        return LLMProvider.OPENAI
    elif "qwen" in model_lower:
        return LLMProvider.QWEN
    elif "aihubmix" in model_lower:
        return LLMProvider.AIHUBMIX
    else:
        return LLMProvider.CUSTOM
