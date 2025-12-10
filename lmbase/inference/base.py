"""
Base Classes and Core Definitions for LLM Inference:

1. API-based
2. Model-based
"""

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod

import torch


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
    messages: Optional[List[Dict[str, Any]]] = None


@dataclass
class InferOutput:
    """
    Standardized output structure for LLM inference responses.

    This dataclass encapsulates the common attributes returned by all LLM
    inference calls, regardless of the provider.

    Attributes:
        prompt (list): The input messages or prompt directly used for the inference request
        response (str): The generated text content from the LLM
        usages (Dict[str, int]): Token usage statistics for the request,
         In general, the usages should include the `time_cost`, `prompt_token_cost`, and `generation_token_cost`.
    """

    prompt: list
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
        generation_config: dict = None,
    ):
        self.lm_name = lm_name
        self.generation_config = generation_config
        self.client = None

    @abstractmethod
    def _initialize_client(self):
        """Initialize the client."""

    @abstractmethod
    def _create_messages(self, infer_input: InferInput, **kwargs) -> Any:
        """Create the messages for the LLM.
        For example, use the ChatPromptTemplate.
        """

    def run(self, infer_input: InferInput, **kwargs) -> InferOutput:
        """Run the synthesizer on the data samples."""

        # convert the input to the target messages required by different APIs.
        messages = (
            infer_input.messages
            if infer_input.messages is not None
            else self._create_messages(infer_input, **kwargs)
        )
        start = time.time()
        output = self._inference(messages)
        output.cost.time_used = time.time() - start
        return output

    @abstractmethod
    def _inference(
        self,
        messages: Any,
    ) -> InferOutput:
        """Synthesize the plans from the data samples."""


@dataclass
class ModelInferOutput(InferOutput):
    """
    Extended output structure for model-based inference.

    This class inherits from `InferOutput` and adds tensor-rich fields commonly
    produced by local/model-based inference (e.g., via PyTorch). It preserves all
    base attributes (`prompt`, `response`, `raw_response`, `cost`, token-level
    fields, and `extras`) while introducing additional vectors and internals for
    downstream analysis.

    Attributes (added):
        input_ids: Tokenized input IDs used by the model.
        generated_ids: Generated token IDs from the model.
        logits: Model output logits for generated tokens.
        hidden_states: Intermediate hidden states across layers.
        attentions: Attention weights across layers/heads.
        embeddings: Final or pooled embedding tensor associated with the output.
    """

    input_ids: Optional[torch.Tensor] = None
    generated_ids: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[List[torch.Tensor]] = None
    attentions: Optional[List[torch.Tensor]] = None
    embeddings: Optional[torch.Tensor] = None


class BaseLMInference(ABC):
    """
    Base class for local/model-based LLM inference.

    Responsibilities:
    - Manage model/tokenizer lifecycle
    - Optionally manage a multimodal `processor` (e.g., vision-language processors)
    - Provide device/dtype configuration
    - Define a standard pipeline: tokenize → model_call → assemble output

    Attributes:
        lm_path: Path or identifier of the local model
        generation_config: Configuration dict used by generation routines. May include
            `device` and `dtype` keys which will be used if provided.
        device: Target device string (e.g., "cuda", "cpu"); can be set via
            `generation_config['device']` and auto-detected if not provided
        dtype: Optional torch dtype used for model weights/inputs; can be set via
            `generation_config['dtype']` or constructor argument. No transformation is performed.
        model: The loaded model instance (set by `_load_model`)
        tokenizer: The loaded tokenizer instance (set by `_load_model`)
        processor: Optional multimodal processor used for images/audio/video
    """

    def __init__(
        self,
        lm_path: str,
        generation_config: dict = None,
    ):
        """
        Initialize the base inference runtime.

        Args:
            lm_path: Path or identifier of the local model.
            generation_config: Configuration dict passed to generation routines.
            **kwargs: Extra configuration passed to subclass implementations.

        Returns:
            None
        """
        self.lm_path = lm_path
        self.generation_config = generation_config or {}

        self.device = (
            self.generation_config["device"]
            if self.generation_config["device"] is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.dtype = (
            self.generation_config["dtype"]
            if "dtype" in self.generation_config
            else None
        )
        self.model = None
        self.tokenizer = None
        self.processor = None
        self._load_model()
        self._load_processor()

    @abstractmethod
    def _load_model(self):
        """
        Load model and tokenizer resources.

        Implementations must set:
            - self.model
            - self.tokenizer

        Returns:
            None
        """
        pass

    def _load_processor(self):
        """
        Optionally load a multimodal processor (e.g., image processors for
        vision-language models).

        Implementations may set:
            - self.processor

        Returns:
            None
        """
        return None

    @abstractmethod
    def _tokenize(self, infer_input: InferInput, **kwargs) -> Dict[str, Any]:
        """
        Convert `InferInput` into model-ready tensors and metadata.

        Args:
            infer_input: Standardized inference input.
            **kwargs: Extra options (e.g., max_length, truncation, image features).

        Returns:
            Dict[str, Any]: Tokenization output ready for `_model_call` (e.g.,
            `input_ids`, `attention_mask`, etc.) placed on `self.device` and
            matching expected dtype when applicable.

        Notes:
            If `self.processor` is available, subclasses may use it to preprocess
            multimodal inputs such as images/audio/video before tokenization.
        """
        pass

    @abstractmethod
    def _model_call(self, infer_input: InferInput, **kwargs) -> ModelInferOutput:
        """
        Execute the entire local/model-based inference pipeline from input to output.

        Expected pipeline:
            - Select messages: use `infer_input.messages` when provided; otherwise
              compose from `infer_input.system_msg` and `infer_input.user_msg`.
            - Preprocess/tokenize: prefer `self.processor` for multimodal inputs;
              fall back to `self.tokenizer` for text-only cases.
            - Device/dtype: move tensors to `self.device`; use `self.dtype` when
              applicable (no conversion performed by the base class).
            - Model execution: perform generation/forward pass and collect vectors
              (e.g., logits, hidden_states, attentions, embeddings).
            - Decode/assemble: produce text outputs and return `ModelInferOutput`.
            - Cost: construct `InferCost`; `run` will set `time_used`.

        Args:
            infer_input: Standardized inference input.
            **kwargs: Extra options (e.g., generation params, return_hidden_states).

        Returns:
            ModelInferOutput
        """
        pass

    def run(self, infer_input: InferInput, **kwargs) -> ModelInferOutput:
        """
        Invoke `_model_call` and populate `cost.time_used`.

        Args:
            infer_input: Standardized inference input.
            **kwargs: Extra options passed through to `_model_call`.

        Returns:
            ModelInferOutput
        """
        start = time.time()
        output = self._model_call(infer_input, **kwargs)
        output.cost.time_used = time.time() - start
        return output
