"""
Implementation of applying the inference based directly on the specific models.

This module provides a concrete visual-language inference class built on the
base interfaces, keeping detailed comments to clarify data flow, tensor shapes,
and outputs for easier understanding and debugging.
"""

from typing import List

import torch
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
)
from vllm import LLM, SamplingParams

from .base import InferInput, ModelInferOutput, InferCost, BaseLMInference


class Qwen25VLInference(BaseLMInference):
    """
    Encapsulate the VLM generation code into a `Qwen25VLInference` class.

    Responsibilities:
    - Load model and processor from `lm_path`
    - Perform end-to-end inference in `_model_call` following the base contract
    - Return `ModelInferOutput` including both text and vector fields
    """

    def _load_model(self):
        # Load a vision-language model using generic AutoModel with remote code.
        # Rationale: many community VLMs expose multimodal generate via `trust_remote_code=True`.
        # If you are explicitly using Qwen2.5-VL, the native class is:
        #   from transformers import Qwen2_5_VLForConditionalGeneration
        #   self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #       self.lm_path, torch_dtype=..., device_map="auto", attn_implementation="eager"
        #   )
        self.model = AutoModel.from_pretrained(
            self.lm_path,
            torch_dtype=self.dtype if self.dtype is not None else "auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.lm_path, trust_remote_code=True
        )

    def _tokenize(self, infer_input: InferInput, **kwargs):
        """
        Convert `InferInput` into processor-ready tensors for Qwen2.5-VL.

        - Build Qwen-style chat `messages` containing both image and text content
        - Use `AutoProcessor.apply_chat_template` to produce the text prompt string
        - Use `process_vision_info` to convert image inputs into RGB tensors
        - Encode with `processor(...)` and move tensors to `self.device`
        """
        if infer_input.messages is not None:
            messages = infer_input.messages
        else:
            # Assume that the user has set the user message correctly as discussed by the comment of the `InferInput`
            messages = [{"role": "user", "content": infer_input.user_msg}]

        # Convert messages into a prompt string following Qwen chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # Extract and preprocess vision inputs (images/videos) into model-ready tensors
        image_inputs, _ = process_vision_info(messages)
        # Encode text and images via the processor; return batched tensors
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        return {"messages": messages, "inputs": inputs}

    def _model_call(
        self,
        infer_inputs: List[InferInput],
        **kwargs,
    ) -> ModelInferOutput:
        """
        Execute the full pipeline from messages → preprocessing/tokenization →
        model generation → decode/assemble outputs.
        """
        tok = self._tokenize(infer_inputs, **kwargs)
        messages = tok["messages"]
        inputs = tok["inputs"]

        # Organize inputs for model inference.
        # 'inputs' is a dict with keys typically including:
        #   - 'input_ids': token IDs of the text prompt, shape [B, L_in]
        #   - 'attention_mask': attention mask for text, shape [B, L_in]
        #   - 'pixel_values': visual features (model-dependent). For image inputs, often
        #       a batch of image tensors in channel-first format. Exact shape depends on
        #       the processor/model config (e.g., [B, C, H, W] or with frames [B, T, C, H, W]).
        #   - 'image_grid_thw': spatial dims after visual encoding, shape [B, 3]
        #       where each row is [T, H', W'] for the i-th sample.
        # Notes:
        #   - B: batch size (here B=1)
        #   - L_in: number of input text tokens
        #   - T: number of frames (images: T=1; videos: T>=1)
        #   - H', W': encoded height/width after the visual backbone

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                # use generation_config for generation parameters (e.g., max_new_tokens, temperature)
                **self.generation_config,
                # return a dict; includes 'sequences' and related generation fields
                return_dict_in_generate=True,
                # also return attentions; structure depends on model configuration
                output_attentions=True,
            )

        # Obtain all model response after reasoning
        # 'sequences': shape [B, L_in + L_gen], includes all input and generated token IDs
        # 'L_gen': length of newly generated tokens
        sequences = outputs.sequences
        input_ids_length = inputs.input_ids.shape[1]
        # 'generated_ids_trimmed': shape [B, L_gen], generated token IDs only
        generated_ids_trimmed = sequences[:, input_ids_length:]
        decoded = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Obtain the IDs of all tokens and convert them into text.
        # 'tokens_text' holds all tokens text and is a list with length = L_in + L_gen
        # Content includes system tokens, visual tokens (if any), prompt tokens,
        # special tokens, and generated tokens.
        seq_ids = outputs.sequences[0].tolist()
        tokens_text = self.processor.tokenizer.convert_ids_to_tokens(
            seq_ids, skip_special_tokens=False
        )

        # Obtain the attention scores for all generated tokens
        # 'attentions' is typically a tuple/list across generation steps, each item
        # may contain per-layer, per-head attention maps. Shapes are model-dependent
        # (e.g., [num_layers, num_heads, S, S] per step where S grows over generation).
        attentions = outputs.attentions
        image_grid_thw = inputs["image_grid_thw"]

        cost = InferCost(time_used=0.0, prompt_tokens=0, completion_tokens=0)
        return ModelInferOutput(
            # 'prompt': original chat messages list for context tracking and reproducibility
            prompt=messages,
            # 'response': final text output (stripped)
            response=decoded.strip(),
            # 'raw_response': decoded raw text (unstripped)
            raw_response=decoded,
            # 'cost': inference cost object (time_used is populated by run())
            cost=cost,
            # 'input_ids': text input token IDs, shape [B, L_in]
            input_ids=inputs.input_ids,
            # 'completion_ids': newly generated token IDs, shape [B, L_gen]
            completion_ids=generated_ids_trimmed,
            # 'logits': optional logits output (None when not enabled)
            logits=None,
            # 'hidden_states': optional hidden states (None when not enabled)
            hidden_states=None,
            # 'attentions': attention weights; structure and dims depend on model configuration
            attentions=list(attentions) if attentions is not None else None,
            # 'embeddings': optional embedding tensor (None when not enabled)
            embeddings=None,
            # 'extras': additional information
            extras={
                # 'tokens_text': list of all token texts, length L_in + L_gen
                "tokens_text": tokens_text,
                # 'image_grid_thw': spatial dims after visual encoding, shape [B, 3] with [T, H', W']
                "image_grid_thw": image_grid_thw,
            },
        )


class LLMInference(BaseLMInference):
    """
    Text-only inference for LLM models with optional vLLM backend.

    Overview:
    - Loads a causal language model and tokenizer from `lm_path`
    - When `use_vllm` is enabled, initializes a vLLM engine for high-throughput generation
    - Implements `_model_call` to perform end-to-end text generation and return `ModelInferOutput`

    Design Notes:
    - Uses the tokenizer's chat template (`apply_chat_template`) to build prompts from messages
    - Accepts direct `**kwargs` for generation parameters (e.g., `max_new_tokens`, `temperature`)
    - Returns `completion_ids` (generated token IDs) alongside text `response`
    - Keeps vector fields (`logits`, `hidden_states`, `attentions`, `embeddings`) disabled by default
    """

    def __init__(
        self,
        lm_path: str,
        inference_config: dict = None,
        generation_config: dict = None,
        **kwargs,
    ):
        super().__init__(
            lm_path=lm_path,
            inference_config=inference_config or {},
            generation_config=generation_config or {},
        )
        self.use_vllm = bool(self.inference_config.get("use_vllm", False))

    def _load_model(self):
        """
        Load tokenizer and model resources.

        - In vLLM mode, initialize `LLM` and keep a HF tokenizer for decoding.
        - Otherwise, use Transformers `AutoModelForCausalLM` and `AutoTokenizer`.
        """
        if self.use_vllm:
            self.model = LLM(model=self.lm_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.lm_path)
            return

        self.model = AutoModelForCausalLM.from_pretrained(
            self.lm_path,
            torch_dtype=self.dtype if self.dtype is not None else "auto",
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.lm_path)

    def _tokenize(self, infer_input: InferInput, **kwargs):
        """
        Convert `InferInput` into text model inputs.

        - Assume user has prepared `user_msg` with the required format and content
        - Build minimal messages and apply the tokenizer's chat template
        - Encode the prompt into tensors (`input_ids`, `attention_mask`) placed on `self.device`
        """
        if infer_input.messages is not None:
            messages = infer_input.messages
        else:
            messages = [{"role": "user", "content": infer_input.user_msg}]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        return {"messages": messages, "inputs": enc, "prompt": prompt}

    def _model_call(
        self,
        infer_input: InferInput,
        **kwargs,
    ) -> ModelInferOutput:
        """
        Execute text generation using either vLLM or Transformers backend.

        - Select backend based on `use_vllm`
        - Pass generation parameters directly via `**kwargs`
        - Decode generated token IDs into text and assemble `ModelInferOutput`
        """
        tok = self._tokenize(infer_input, **kwargs)
        messages = tok["messages"]
        inputs = tok["inputs"]

        if self.use_vllm and self.model is not None:
            # Build sampling parameters directly from **kwargs so callers can control
            # decoding behavior (e.g., max_new_tokens, temperature, top_p, top_k).
            prompt_str = tok["prompt"]
            sampling = SamplingParams(**self.generation_config)
            # Generate using vLLM engine; returns structured outputs per prompt
            results = self.model.generate(prompt_str, sampling_params=sampling)

            # Extract the first candidate from the first prompt for simplicity
            first = results[0].outputs[0]
            decoded = first.text

            # Convert vLLM token IDs to a 2D tensor [1, L_gen] to match ModelInferOutput
            token_ids = getattr(first, "token_ids", None)
            completion_ids = (
                torch.tensor(token_ids, device=self.device).unsqueeze(0)
                if token_ids is not None
                else None
            )

            # Optional token text reconstruction for debugging/analysis
            tokens_text = (
                self.tokenizer.convert_ids_to_tokens(
                    token_ids, skip_special_tokens=False
                )
                if token_ids is not None
                else None
            )
            cost = InferCost(time_used=0.0, prompt_tokens=0, completion_tokens=0)
            return ModelInferOutput(
                prompt=messages,
                response=decoded.strip(),
                raw_response=decoded,
                cost=cost,
                input_ids=inputs.input_ids,
                completion_ids=completion_ids,
                logits=None,
                hidden_states=None,
                attentions=None,
                embeddings=None,
                extras={"tokens_text": tokens_text},
            )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.generation_config,
                return_dict_in_generate=True,
                output_attentions=True,
            )
        # Collect sequences ([B, L_in + L_gen]) and slice out generated IDs
        sequences = outputs.sequences
        input_len = inputs.input_ids.shape[1]
        completion_ids = sequences[:, input_len:]

        # Decode generated token IDs to text; keep special tokens skipped
        decoded = self.tokenizer.batch_decode(
            completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Build token text for full sequence (input + completion) when needed
        seq_ids = sequences[0].tolist()
        tokens_text = self.tokenizer.convert_ids_to_tokens(
            seq_ids, skip_special_tokens=False
        )

        # Capture attentions returned by generate; structure depends on model config
        attentions = outputs.attentions

        cost = InferCost(time_used=0.0, prompt_tokens=0, completion_tokens=0)
        return ModelInferOutput(
            prompt=messages,
            response=decoded.strip(),
            raw_response=decoded,
            cost=cost,
            input_ids=inputs.input_ids,
            completion_ids=completion_ids,
            logits=None,
            hidden_states=None,
            attentions=list(attentions) if attentions is not None else None,
            embeddings=None,
            extras={"tokens_text": tokens_text},
        )
