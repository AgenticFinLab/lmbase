"""LLM client builder for unified provider selection."""

import os
import random
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional
from langchain_openai import ChatOpenAI


def aihubmix_api_call(prompt: str, model_id: str) -> str:

    load_dotenv()
    AIHUBMIX_API_KEY = os.getenv("AIHUBMIX_API_KEY")
    AIHUBMIX_BASE_URL = os.getenv("AIHUBMIX_BASE_URL")

    client = OpenAI(
        base_url=AIHUBMIX_BASE_URL,
        api_key=AIHUBMIX_API_KEY,
    )

    completion = client.chat.completions.create(
        model=model_id,  # model ID, copy from https://aihubmix.com/models
        messages=[
            # {"role": "developer", "content": ""},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        seed=random.randint(1, 1000000000),
    )

    return completion.choices[0].message.content


def build_llm(
    model_override: Optional[str] = None,
    temperature: float = 0,
) -> ChatOpenAI:
    """Construct a ChatOpenAI client with code-level defaults (no env for provider/model).

    Selection rules:
    - If `model_override` looks like a Doubao model (prefix `doubao-`), use Doubao compatible endpoint.
    - If `model_override` contains `deepseek`, use DeepSeek compatible endpoint.
    - If `model_override` looks like an OpenAI model (contains `gpt`), use OpenAI default endpoint.
    - Otherwise default to Qwen compatible endpoint with a general model.

    Notes:
    - Only API keys may be read from environment when not provided by the runtime.
    - Default Qwen base URL: `https://dashscope.aliyuncs.com/compatible-mode/v1`
    - Default Doubao base URL: `https://ark.cn-beijing.volces.com/api/v3`
    - Default DeepSeek base URL: `https://api.deepseek.com`
    """
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

    if model_override and "deepseek" in model_override.lower():
        base_url = "https://api.deepseek.com"
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        return ChatOpenAI(
            model=model_override,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
        )

    if model_override and ("gpt" in model_override.lower()):
        api_key = os.getenv("OPENAI_API_KEY")
        return ChatOpenAI(
            model=model_override, temperature=temperature, api_key=api_key
        )

    # default: qwen
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = model_override
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )
