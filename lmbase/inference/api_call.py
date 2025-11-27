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

    def __init__(
        self, lm_name=None, base_url=None, api_key=None, generation_config=None
    ):
        super().__init__(
            lm_name=lm_name,
            base_url=base_url,
            api_key=api_key,
            generation_config=generation_config,
        )

    def _initialize_client(self):
        self.client = ChatOpenAI(
            model=self.lm_name, base_url=self.base_url, api_key=self.api_key
        )

    def _create_messages(
        self,
        infer_input: InferInput,
        **kwargs,
    ) -> ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_messages(
            [("system", "{system_msg}"), ("human", "{user_msg}")]
        )
        return prompt.format_messages(
            system_msg=infer_input.system_msg, user_msg=infer_input.user_msg
        )

    def _inference(
        self,
        messages: list,
    ) -> InferOutput:
        """Synthesize the plans from the data samples."""
        response = self.client.invoke(messages)

        return InferOutput(
            prompt=messages,
            response=response.content,
            raw_response=response,
            cost=InferCost(
                time_used=None,
                prompt_tokens=response.usage_metadata["input_tokens"],
                completion_tokens=response.usage_metadata["input_tokens"],
            ),
        )
