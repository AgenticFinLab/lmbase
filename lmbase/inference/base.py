"""
Base Classes and Core Definitions for LLM Inference.

We support two methods:

1. API-based
2. Model-based

"""

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod


@dataclass
class InferCost:
    """
    Cost of the inference.
    """

    time_used: float
    prompt_tokens: int
    completion_tokens: int

    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferInput:
    """
    Standardized input structure for LLM inference requests.

    This dataclass encapsulates the common parameters required for all LLM
    inference calls, regardless of the provider.

    Attributes:
        system_msg: The input message used for the system prompt.
        user_msg: The input message used for the user prompt.
        extras: Additional information to be used.
    """

    system_msg: str
    user_msg: str

    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferOutput:
    """
    Standardized output structure for LLM inference responses.

    This dataclass encapsulates the common attributes returned by all LLM
    inference calls, regardless of the provider.

    Attributes:
        prompt (str): The input prompt directly used for the inference request
        response (str): The generated text content from the LLM
        usages (Dict[str, int]): Token usage statistics for the request,
         In general, the usages should include the `time_cost`, `prompt_token_cost`, and `generation_token_cost`.
    """

    prompt: str
    response: str
    raw_response: str

    cost: InferCost

    prompt_tokens: Optional[List[str]] = None
    response_tokens: Optional[List[str]] = None
    raw_response_tokens: Optional[List[str]] = None

    extras: Dict[str, Any] = field(default_factory=dict)


class BaseLMAPIInference(ABC):
    """
    Base class for the large model inference based on the APIs.
    """

    def __init__(
        self,
        lm_name: str,
        generation_config: dict,
    ):
        self.lm_name = lm_name
        self.generation_config = generation_config

        # In general, most API platforms requires the creation of a client.
        # In special cases, such as the langchain, we should define a term such as ChatOpenAI and then invoke it for the response.
        # For simplicity, we use the consistent term client.
        self.client = None

    @abstractmethod
    def initialize_client(self, **kwargs):
        """Initialize the client."""

    @abstractmethod
    def create_messages(self, infer_input: InferInput, **kwargs) -> Any:
        """Create the messages for the LLM.
        For example, use the ChatPromptTemplate.
        """

    @abstractmethod
    def inference(
        self,
        messages: Any,
    ) -> InferOutput:
        """Synthesize the plans from the data samples."""

    def run(self, infer_input: InferInput, **kwargs) -> InferOutput:
        """Run the synthesizer on the data samples."""

        # convert the input to the target messages required by different APIs.
        messages = self.create_messages(infer_input, **kwargs)
        start = time.time()
        output = self.inference(messages, **kwargs)
        output.cost.time_used = time.time() - start
        return output
