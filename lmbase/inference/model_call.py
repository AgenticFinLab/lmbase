"""
Implementation of applying the inference based directly on the specific models.

This module provides a concrete visual-language inference class built on the
base interfaces, keeping detailed comments to clarify data flow, tensor shapes,
and outputs for easier understanding and debugging.
"""

import torch
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

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
        # Load Qwen2.5-VL model from the specified path
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.lm_path,
            torch_dtype=self.dtype if self.dtype is not None else "auto",
            device_map="auto",
            attn_implementation="eager",
        )
        self.tokenizer = None

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(self.lm_path)

    def _model_call(self, infer_input: InferInput, **kwargs) -> ModelInferOutput:
        """
        Execute the full pipeline from messages → preprocessing/tokenization →
        model generation → decode/assemble outputs.
        """
        if infer_input.messages is not None:
            messages = infer_input.messages
        else:
            image_path = (
                infer_input.extras["image_path"]
                if "image_path" in infer_input.extras
                else None
            )
            user_text = infer_input.user_msg
            content = []
            if image_path is not None:
                content.append({"type": "image", "image": image_path})
            if user_text is not None:
                content.append({"type": "text", "text": user_text})
            messages = [{"role": "user", "content": content}]

        # Process image and text inputs to meet the template requirements of the model
        # Use processor chat template (Qwen-specific)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process the input image (e.g., an image URL) to obtain the image data in RGB mode
        image_inputs, _ = process_vision_info(messages)

        # Build processor kwargs dynamically to include images only when present
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

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

        gen_kwargs = {}
        for k, v in (
            self.generation_config.items()
            if self.generation_config is not None
            else {}.items()
        ):
            gen_kwargs[k] = v
        for k, v in kwargs.items():
            gen_kwargs[k] = v

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs,
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
            # 'generated_ids': newly generated token IDs, shape [B, L_gen]
            generated_ids=generated_ids_trimmed,
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
