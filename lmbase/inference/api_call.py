"""
LLM Client Builder for Unified Provider Selection

This module provides a unified interface for building LLM clients across multiple AI service providers including AIHubMix, Doubao, DeepSeek, OpenAI, and Qwen. It handles provider-specific configuration, authentication, and client initialization.

The module uses OpenAI-compatible client for direct API calls without LangChain dependency.
"""

import os
from typing import List, Dict, Any

from openai import OpenAI

from lmbase.inference.base import InferCost, InferInput, InferOutput, BaseLMAPIInference


class LangChainAPIInference(BaseLMAPIInference):
    """
    API-based LLM inference implementation using OpenAI-compatible client.

    This class provides access to multiple LLM providers (AIHubMix, Doubao, DeepSeek, OpenAI, Qwen)
    through OpenAI-compatible API endpoints. The class name is retained for backward compatibility.

    Note: Despite the name, this implementation uses the OpenAI client directly, not LangChain.
    """

    def __init__(
        self,
        lm_name=None,
        generation_config=None,
    ):
        super().__init__(lm_name=lm_name, generation_config=generation_config)
        base_urls = {
            "ark": "https://ark.cn-beijing.volces.com/api/v3",
            "deepseek": "https://api.deepseek.com/v1",
            "openai": "https://api.openai.com/v1",
            "qwen": "https://ark.cn-beijing.volces.com/api/v3",
            "aihubmix": "https://aihubmix.com/v1",
        }
        self.lm_provider = lm_name.split("/")[0]
        self.model_name = "/".join(lm_name.split("/")[1:])
        self.base_url = base_urls[self.lm_provider.lower()]
        self.api_key = os.getenv(f"{self.lm_provider.upper()}_API_KEY")
        self.max_tokens = 8192
        if not self.api_key or self.api_key is None:
            raise ValueError(
                f"API key for provider '{self.lm_provider.upper()}' not found. Please set the environment variable '{self.lm_provider.upper()}_API_KEY'."
            )
        self._initialize_client()

    def _initialize_client(self):
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def _create_messages(
        self,
        infer_input: InferInput,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Create messages in OpenAI format."""
        messages = []

        # Format system message
        if infer_input.system_msg:
            system_msg = (
                infer_input.system_msg.format(**kwargs)
                if kwargs
                else infer_input.system_msg
            )
            messages.append({"role": "system", "content": system_msg})

        # Format user message
        user_msg = infer_input.user_msg
        if isinstance(user_msg, str):
            user_content = user_msg.format(**kwargs) if kwargs else user_msg
            messages.append({"role": "user", "content": user_content})
        elif isinstance(user_msg, list):
            # Handle multimodal input (e.g., for vision models)
            # Format each item in the list if it's a string template
            formatted_content = []
            for item in user_msg:
                if isinstance(item, dict):
                    # Create a copy to avoid modifying the original
                    item_copy = item.copy()
                    if "text" in item_copy and kwargs:
                        item_copy["text"] = item_copy["text"].format(**kwargs)
                    formatted_content.append(item_copy)
                elif isinstance(item, str):
                    formatted_content.append(item.format(**kwargs) if kwargs else item)
                else:
                    formatted_content.append(item)
            messages.append({"role": "user", "content": formatted_content})
        else:
            messages.append({"role": "user", "content": user_msg})

        return messages

    def _inference(
        self,
        messages: List[Dict[str, Any]],
    ) -> InferOutput:
        """Synthesize the plans from the data samples."""
        # Prepare generation config
        generation_params = {
            "model": self.model_name,
            "messages": messages,
        }

        # Merge with generation_config if provided
        if self.generation_config:
            generation_params.update(self.generation_config)
        else:
            # Set default max_tokens if not in generation_config
            generation_params["max_tokens"] = self.max_tokens

        # Call OpenAI API
        response = self.client.chat.completions.create(**generation_params)

        # Extract response content
        response_content = response.choices[0].message.content
        if response_content is None:
            response_content = ""

        # Extract usage information
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        return InferOutput(
            prompt=messages,
            response=response_content.strip(),
            raw_response=response_content,
            cost=InferCost(
                time_used=None,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
        )
