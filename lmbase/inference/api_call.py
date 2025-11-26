"""
LLM Client Builder for Unified Provider Selection

This module provides a unified interface for building LLM clients across multiple
AI service providers including AIHubMix, Doubao, DeepSeek, OpenAI, and Qwen.
It handles provider-specific configuration, authentication, and client initialization.

The module supports both direct OpenAI client usage and LangChain ChatOpenAI integration
for different use cases in AI application development.
"""

import os
import random
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any, Union
from langchain_openai import ChatOpenAI

from lmbase.inference.base import (
    LLMProvider,
    Message,
    LLMRequest,
    LLMResponse,
    BaseLLMClient,
    detect_provider,
)


class AIHubMixClient(BaseLLMClient):
    """
    AIHubMix-specific LLM client implementation.

    AIHubMix is a unified API platform that provides access to multiple LLM providers
    through a single interface.
    """

    def initialize_client(self) -> OpenAI:
        """Initialize and return the OpenAI client configured for AIHubMix."""
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        return self.client

    def call(self, request: LLMRequest) -> LLMResponse:
        """
        Execute an LLM inference call through AIHubMix.

        Args:
            request: Standardized LLM request parameters

        Returns:
            Standardized LLM response

        Raises:
            Exception: For API communication errors or authentication failures
        """
        self.validate_request(request)

        if self.client is None:
            self.initialize_client()

        # Convert to API format and ensure seed is set
        api_params = request.to_api_format()
        if api_params.get("seed") is None:
            api_params["seed"] = random.randint(1, 1000000000)

        # Execute chat completion
        completion = self.client.chat.completions.create(**api_params)

        # Extract response content
        content = completion.choices[0].message.content
        if content is None:
            content = ""

        # Build standardized response
        usage_dict = {}
        if completion.usage:
            usage_dict = {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            }

        return LLMResponse(
            content=content,
            model=completion.model,
            usage=usage_dict,
            finish_reason=completion.choices[0].finish_reason or "stop",
            raw_response=completion,
        )


def build_llm_aihubmix(
    model_id: str,
    prompt: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    seed: Optional[int] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    return_client_only: bool = False,
) -> Union[AIHubMixClient, str]:
    """
    Build and configure an AIHubMix LLM client with flexible usage options.

    AIHubMix is a unified API platform that provides access to multiple LLM providers
    through a single interface. This function can return either a configured client
    or directly execute a completion request.

    AIHubMix Model List: https://aihubmix.com/models
    AIHubMix Documentation: https://docs.aihubmix.com/cn

    Args:
        model_id (str): The specific model ID to use from AIHubMix model catalog.
                       Required for all operations.
        prompt (Optional[str]): The input prompt text. Required if not using messages
                               parameter and not returning client only.
        temperature (float): Controls randomness in generation (0.0 to 1.0).
                           Lower values make output more deterministic.
        max_tokens (int): Maximum number of tokens to generate in the response.
        top_p (float): Nucleus sampling parameter (0.0 to 1.0).
        frequency_penalty (float): Penalizes frequently used tokens (-2.0 to 2.0).
        presence_penalty (float): Penalizes new token presence (-2.0 to 2.0).
        seed (Optional[int]): Random seed for reproducible results. Auto-generated if None.
        messages (Optional[List[Dict]]): Pre-formatted message list for chat completion.
                                  If provided, overrides prompt parameter.
        return_client_only (bool): If True, returns configured client instead of
                                  executing completion.

    Returns:
        Union[AIHubMixClient, str]:
            - If return_client_only=True: Configured AIHubMixClient instance
            - Otherwise: Generated text response as string

    Raises:
        ValueError: If required parameters are missing for completion request
        Exception: For API communication errors or authentication failures

    Example:
        # Get client only
        client = build_llm_aihubmix("gpt-4", return_client_only=True)

        # Direct completion
        response = build_llm_aihubmix("gpt-4", prompt="Hello world")

        # Custom messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain AI."}
        ]
        response = build_llm_aihubmix("gpt-4", messages=messages)
    """
    # Load environment variables from .env file
    load_dotenv()

    # Retrieve AIHubMix API credentials from environment variables
    AIHUBMIX_API_KEY = os.getenv("AIHUBMIX_API_KEY")
    AIHUBMIX_BASE_URL = os.getenv("AIHUBMIX_BASE_URL")

    # Validate required environment variables
    if not AIHUBMIX_API_KEY:
        raise ValueError("AIHUBMIX_API_KEY environment variable is required")
    if not AIHUBMIX_BASE_URL:
        raise ValueError("AIHUBMIX_BASE_URL environment variable is required")

    # Create AIHubMix client
    client = AIHubMixClient(api_key=AIHUBMIX_API_KEY, base_url=AIHUBMIX_BASE_URL)
    client.initialize_client()

    # If only client is requested, return the configured client
    if return_client_only:
        return client

    # Prepare messages for completion request
    if messages is None:
        # Validate that we have content for completion
        if not prompt:
            raise ValueError(
                "Either prompt or messages parameter must be provided for completion request"
            )

        # Create messages list from prompt
        message_objs = [Message(role="user", content=prompt)]
    else:
        # Convert dictionary messages to Message objects
        message_objs = [
            Message(role=msg["role"], content=msg["content"]) for msg in messages
        ]

    # Create standardized request
    request = LLMRequest(
        model=model_id,
        messages=message_objs,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        seed=seed,
    )

    # Execute the call and return content
    response = client.call(request)
    return response.content


def build_llm(
    model_override: Optional[str] = None,
    temperature: float = 0,
) -> ChatOpenAI:
    """
    Construct a ChatOpenAI client with code-level defaults for multiple providers.

    This function provides a unified interface for creating LangChain ChatOpenAI
    clients configured for different AI service providers based on model naming patterns.

    Selection Rules:
    - Doubao models: Identified by 'doubao' in model name, uses VolcEngine ARK endpoint
    - DeepSeek models: Identified by 'deepseek' in model name, uses DeepSeek endpoint
    - OpenAI models: Identified by 'gpt' in model name, uses standard OpenAI endpoint
    - Default: Qwen compatible endpoint as fallback

    Args:
        model_override (Optional[str]): Specific model identifier. Provider is
                                       auto-detected from model name patterns.
        temperature (float): Sampling temperature for generation (0.0 to 1.0).
                           Lower values produce more deterministic outputs.

    Returns:
        ChatOpenAI: Configured LangChain ChatOpenAI client instance

    Environment Variables:
        DOUBAO_API_KEY, ARK_API_KEY: For Doubao/VolcEngine ARK authentication
        DEEPSEEK_API_KEY: For DeepSeek service authentication
        OPENAI_API_KEY: For OpenAI service or as fallback authentication
        DASHSCOPE_API_KEY: For Qwen/DashScope service authentication

    Example:
        # Doubao model
        doubao_llm = build_llm("doubao-7b", temperature=0.1)

        # DeepSeek model
        deepseek_llm = build_llm("deepseek-coder", temperature=0.3)

        # OpenAI model
        openai_llm = build_llm("gpt-4", temperature=0.7)

        # Default Qwen model
        qwen_llm = build_llm("qwen-plus", temperature=0.5)
    """

    # Provider detection and configuration for Doubao models
    if model_override and "doubao" in model_override.lower():
        base_url = "https://ark.cn-beijing.volces.com/api/v3"
        api_key = (
            os.getenv("DOUBAO_API_KEY")
            or os.getenv("ARK_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        return ChatOpenAI(
            model=model_override,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
        )

    # Provider detection and configuration for DeepSeek models
    if model_override and "deepseek" in model_override.lower():
        base_url = "https://api.deepseek.com"
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        return ChatOpenAI(
            model=model_override,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
        )

    # Provider detection and configuration for OpenAI models
    if model_override and ("gpt" in model_override.lower()):
        api_key = os.getenv("OPENAI_API_KEY")
        return ChatOpenAI(
            model=model_override, temperature=temperature, api_key=api_key
        )

    # Default provider configuration (Qwen/DashScope)
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = model_override

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )
