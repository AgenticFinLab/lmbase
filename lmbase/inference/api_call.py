"""
LLM Client Builder for Unified Provider Selection

This module provides a unified interface for building LLM clients across multiple AI service providers including AIHubMix, Doubao, DeepSeek, OpenAI, and Qwen. It handles provider-specific configuration, authentication, and client initialization.

The module supports both direct OpenAI client usage and LangChain ChatOpenAI integration for different use cases in AI application development.
"""

import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage

from lmbase.inference.base import InferCost, InferInput, InferOutput, BaseLMAPIInference


class LangChainAPIInference(BaseLMAPIInference):
    """
    LangChain-specific LLM inference implementation.

    LangChain is a unified API platform that provides access to multiple LLM providers through a single interface.
    """

    def __init__(
        self,
        lm_provider=None,
        lm_name=None,
        generation_config=None,
    ):
        super().__init__(lm_name=lm_name, generation_config=generation_config)
        base_urls = {
            "doubao": "https://ark.cn-beijing.volces.com/api/v3",
            "deepseek": "https://api.deepseek.com/v1",
            "openai": "https://api.openai.com/v1",
            "qwen": "https://ark.cn-beijing.volces.com/api/v3",
            "aihubmix": "https://api.aihubmix.com/v1",
        }
        self.base_url = base_urls[lm_provider.lower()]
        self.api_key = os.getenv(f"{lm_provider.upper()}_API_KEY")
        self._initialize_client()

    def _initialize_client(self):
        self.client = ChatOpenAI(
            model=self.lm_name,
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def _create_messages(
        self,
        infer_input: InferInput,
        **kwargs,
    ) -> list[BaseMessage]:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", infer_input.system_msg),
                ("human", infer_input.user_msg),
            ]
        )
        return prompt.format_messages(**kwargs)

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
