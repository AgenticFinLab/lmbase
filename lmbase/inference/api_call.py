"""
LLM Client Builder for Unified Provider Selection

This module provides a unified interface for building LLM clients across multiple
AI service providers including AIHubMix, Doubao, DeepSeek, OpenAI, and Qwen.
It handles provider-specific configuration, authentication, and client initialization.

The module supports both direct OpenAI client usage and LangChain ChatOpenAI integration
for different use cases in AI application development.
"""

import os
from typing import Any

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from lmbase.inference.base import InferCost, InferInput, InferOutput, BaseLMAPIInference


class LangChainAPIInference(BaseLMAPIInference):
    """
    LangChain-specific LLM inference implementation.

    LangChain is a unified API platform that provides access to multiple LLM providers through a single interface.
    """

    def __init__(self, lm_name, generation_config):
        super().__init__(lm_name, generation_config)

        base_urls = {
            "doubao": "https://ark.cn-beijing.volces.com/api/v3",
            "deepseek": "https://api.deepseek.cn/v1",
            "openai": "https://api.openai.com/v1",
            "qwen": "https://ark.cn-beijing.volces.com/api/v3",
        }
        model_type = self.lm_name.split("-")[0].lower()
        base_model = "OPENAI" if "gpt" in model_type else model_type
        self.base_url = base_urls[model_type.lower()]
        self.api_key = os.getenv(f"{base_model.upper()}_API_KEY")

    def initialize_client(self, **kwargs):
        self.client = ChatOpenAI(
            model=self.lm_name,
            base_url=self.base_url,
            api_key=self.api_key,
            **self.generation_config,
        )

    def create_messages(
        self,
        infer_input: InferInput,
        **kwargs,
    ) -> ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", infer_input.system_msg),
                ("human", infer_input.user_msg),
            ]
        )

        return prompt.format_messages(**kwargs)

    def inference(
        self,
        messages: ChatPromptTemplate,
    ) -> InferOutput:
        """Synthesize the plans from the data samples."""
        response = self.client.invoke(messages)

        return InferOutput(
            prompt=messages.format(),
            response=response.content,
            raw_response=response,
            cost=InferCost(
                time_used=None,
                prompt_tokens=response.usage_metadata["input_tokens"],
                completion_tokens=response.usage_metadata["input_tokens"],
            ),
        )


class AIHubMixAPIInference(BaseLMAPIInference):
    def __init__(self, lm_name, generation_config):
        super().__init__(lm_name, generation_config)
        self.base_url = os.getenv("AIHUBMIX_BASE_URL") or ""
        self.api_key = os.getenv("AIHUBMIX_API_KEY") or ""

    def initialize_client(self, **kwargs):
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def create_messages(
        self,
        infer_input: InferInput,
        **kwargs,
    ) -> Any:
        return [
            {"role": "system", "content": infer_input.system_msg},
            {"role": "user", "content": infer_input.user_msg},
        ]

    def inference(
        self,
        messages: Any,
    ) -> InferOutput:

        completion = self.client.chat.completions.create(
            model=self.lm_name,
            messages=messages,
            **self.generation_config,
        )
        content = completion.choices[0].message.content or ""
        usage = completion.usage
        prompt_text = "\n".join([m["content"] for m in messages])

        return InferOutput(
            prompt=prompt_text,
            response=content,
            raw_response=str(completion),
            cost=InferCost(
                time_used=None,
                prompt_tokens=(usage.prompt_tokens if usage else 0),
                completion_tokens=(usage.completion_tokens if usage else 0),
            ),
        )
