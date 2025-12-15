"""
Base Classes and Core Definitions for LLM Inference:

1. API-based
2. Model-based
"""

import json
import time
from dataclasses import dataclass, field, asdict
import dataclasses

from typing import Optional, List, Dict, Any, Union
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
            Supported forms vary by model type:
            - Text-only models: pass a plain `str` containing the user question or instruction.
            - Vision-language (VL) models: pass either a `str` or a `List` of content blocks.
              When passing a list, use Qwen-style content items, for example:
                [{"type": "image", "image": "/path/to/image.jpg"},
                 {"type": "text",  "text":  "Describe the image"}]
              The list may contain multiple images and text segments in order.

            Notes:
            - If `messages` is provided, `user_msg` is ignored; `messages` takes precedence.
            - For VL models, always prefer explicit list-form content for multi-modal prompts.
        extras: Additional information to be used.
    """

    system_msg: str
    user_msg: Union[str, List[Any]]

    extras: Dict[str, Any] = field(default_factory=dict)
    messages: Optional[List[Dict[str, Any]]] = None


@dataclass
class InferOutput:
    """
    Standardized output structure for LLM inference responses.

    Overview:
    - Encapsulates the common attributes returned by local or API-based inference
    - Designed to be serializable via `to_dict()` even when fields include complex objects

    Field Details:
    - `prompt` (list):
      The messages or prompt content used for the request. This can be:
        - A list of plain dicts (e.g., `[{"role": "user", "content": "..."}]`)
        - A list of framework-specific message objects (e.g., LangChain `HumanMessage`),
          which `to_dict()` will recursively convert to serializable structures.

    - `response` (str):
      The final decoded text content produced by the model, typically trimmed of leading/trailing whitespace.

    - `raw_response` (str):
      The unmodified decoded text content prior to any stripping or cleanup. Useful for debugging or exact output reproduction.

    - `cost` (InferCost):
      Aggregated timing and token accounting information. `run()` populates `cost.time_used` after inference.

    - `prompt_tokens` (Optional[List[str]]):
      The textual representation of tokens belonging to the input prompt when available. Useful for debugging tokenization.

    - `response_tokens` (Optional[List[str]]):
      The textual representation of tokens belonging solely to the generated completion when available.

    - `raw_response_tokens` (Optional[List[str]]):
      The textual representation of the entire decoded output (including special tokens) without cleanup, when available.

    - `extras` (Dict[str, Any]):
      A catch-all for additional metadata. Examples:
        - `{"tokens_text": [...]}`: full token texts for input + completion
        - `{"image_grid_thw": [T, H', W']}`: visual backbone grid dims for multimodal models
        - Any other per-model diagnostic or analysis information

    Serialization:
    - Use `to_dict()` to obtain a nested, JSON-friendly representation.
      Complex objects (dataclasses, tensors, message classes) are converted recursively.
    """

    prompt: list
    response: str
    raw_response: str

    cost: InferCost

    prompt_tokens: Optional[List[str]] = None
    response_tokens: Optional[List[str]] = None
    raw_response_tokens: Optional[List[str]] = None

    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        """
        Convert this output into a JSON-friendly dict by recursively visiting
        all fields and stringifying anything that is not directly serializable.
        """

        def _ser(obj):
            if obj is None:
                return None
            if isinstance(obj, (str, int, float, bool)):
                return obj
            if dataclasses.is_dataclass(obj):
                return _ser(asdict(obj))
            if isinstance(obj, dict):
                return {str(k): _ser(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple, set)):
                return [_ser(v) for v in obj]
            if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
                try:
                    return _ser(obj.to_dict())
                except Exception:
                    return str(obj)
            if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
                try:
                    return _ser(obj.dict())
                except Exception:
                    return str(obj)
            if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
                try:
                    return _ser(obj.model_dump())
                except Exception:
                    return str(obj)
            if hasattr(obj, "to_json") and callable(getattr(obj, "to_json")):
                try:
                    j = obj.to_json()
                    if isinstance(j, str):
                        return json.loads(j)
                    return _ser(j)
                except Exception:
                    return str(obj)
            return str(obj)

        return _ser(self)


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
        completion_ids: Generated token IDs from the model.
        logits: Model output logits for generated tokens.
        hidden_states: Intermediate hidden states across layers.
        attentions: Attention weights across layers/heads.
        embeddings: Final or pooled embedding tensor associated with the output.
    """

    input_ids: Optional[torch.Tensor] = None
    completion_ids: Optional[torch.Tensor] = None
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
        inference_config: Non-generation configuration for inference runtime (e.g., `device`, `dtype`,
            backend toggles like `use_vllm`, etc.).
        generation_config: Configuration dict passed directly to model `generate` routines (e.g.,
            `max_new_tokens`, `temperature`, `top_p`).
        device: Target device string (e.g., "cuda", "cpu"); controlled by `inference_config['device']`
            and auto-detected if not provided.
        dtype: Optional torch dtype used for model weights/inputs; controlled by `inference_config['dtype']`.
        model: The loaded model instance (set by `_load_model`)
        tokenizer: The loaded tokenizer instance (set by `_load_model`)
        processor: Optional multimodal processor used for images/audio/video
    """

    def __init__(
        self,
        lm_path: str,
        inference_config: dict = None,
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
        # Separate configs:
        # - inference_config: non-generation runtime settings (device, dtype, backend toggles, etc.)
        # - generation_config: parameters passed to the model's `generate` API
        self.inference_config = inference_config or {}
        self.generation_config = generation_config or {}

        self.device = (
            self.inference_config["device"]
            if self.inference_config.get("device") is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.dtype = (
            self.inference_config["dtype"] if "dtype" in self.inference_config else None
        )
        self.model = None
        self.tokenizer = None
        self.processor = None
        self._load_model()

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

    @abstractmethod
    def _tokenize(self, infer_inputs: List[InferInput], **kwargs) -> Dict[str, Any]:
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
    def _model_call(self, infer_inputs: List[InferInput], **kwargs) -> ModelInferOutput:
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

    def run(self, infer_inputs: List[InferInput], **kwargs) -> ModelInferOutput:
        """
        Invoke `_model_call` and populate `cost.time_used`.

        Args:
            infer_inputs: Standardized inference inputs for a batch.
            **kwargs: Extra options passed through to `_model_call`.

        Returns:
            ModelInferOutput
        """
        start = time.time()
        output = self._model_call(infer_inputs, **kwargs)
        output.cost.time_used = time.time() - start
        return output
