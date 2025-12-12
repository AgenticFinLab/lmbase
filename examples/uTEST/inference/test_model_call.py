import os
from lmbase.inference.base import InferInput
from lmbase.inference.model_call import Qwen25VLInference, LLMInference


# This demo file shows:
# 1) How to run text-only LLM inference (LLMInference) with inference_config and generation_config
# 2) How to run vision-language inference (Qwen25VLInference) with Qwen-style content list
#
# Environment variables:
# - LLM_PATH: path or hub name for the text model (default: Qwen/Qwen2.5-7B-Instruct)
# - VLM_PATH: path or hub name for the VLM (default: Qwen/Qwen2.5-VL-7B-Instruct)
#
# Config conventions:
# - inference_config: non-generation runtime settings (e.g., device, dtype, backend toggles like use_vllm)
# - generation_config: parameters passed directly to model generate (e.g., max_new_tokens, temperature, top_p)


def demo_llm():
    # Select an LLM path; can be a local directory or a HF Hub name
    lm_path = os.getenv("LLM_PATH", "Qwen/Qwen2.5-7B-Instruct")

    # Text inference: disable vLLM and use Transformers backend
    # inference_config controls non-generation runtime settings (use_vllm=False here)
    infer = LLMInference(lm_path=lm_path, inference_config={"use_vllm": False})

    # Build a standardized input; text models typically use a string user_msg
    inp = InferInput(system_msg="", user_msg="Write a short greeting.")

    try:
        # You can pass generation params via **kwargs or preset infer.generation_config
        # Since internals read from generation_config, the following also works:
        #   infer.generation_config.update({"max_new_tokens": 16, "temperature": 0.7})
        out = infer.run(inp, max_new_tokens=16, temperature=0.7)

        # Output structure:
        # - out.response: final text stripped
        # - out.raw_response: raw decoded text
        # - out.input_ids: input token IDs, shape [B, L_in]
        # - out.completion_ids: generated token IDs, shape [B, L_gen]
        # - out.attentions: attention weights (when output_attentions is enabled)
        print("LLM response:", out.response)
    except Exception as e:
        # Simple error reporting; in production, handle specific exceptions accordingly
        print("LLM demo failed:", e)


def demo_vlm():
    # Select a VLM path; can be local or a HF Hub name
    vlm_path = os.getenv("VLM_PATH", "Qwen/Qwen2.5-VL-7B-Instruct")
    infer = Qwen25VLInference(lm_path=vlm_path)

    # Qwen-style content list (multimodal input):
    # - Each item is a content block, type can be "image" or "text"
    # - For image items, "image" points to a path or processor-recognized resource
    # - This demo uses text-only; to add image: {"type": "image", "image": "/path/to.jpg"}
    content = [{"type": "text", "text": "Describe the provided content briefly."}]
    inp = InferInput(system_msg="", user_msg=content)

    try:
        # Put generation parameters into generation_config so the inference reads them uniformly
        infer.generation_config.update({"max_new_tokens": 16, "temperature": 0.7})
        out = infer.run(inp)

        # Output structure:
        # - out.response / out.raw_response: decoded text via processor.batch_decode
        # - out.input_ids: text input token IDs
        # - out.completion_ids: generated token IDs (text only portion)
        # - out.extras["image_grid_thw"]: visual encoding dims [T, H', W']
        # - out.extras["tokens_text"]: token texts for the full sequence (may include special/vision tokens)
        print("VLM response:", out.response)
    except Exception as e:
        print("VLM demo failed:", e)


if __name__ == "__main__":
    # Run this file directly to see both text and multimodal inference flows
    demo_llm()
    demo_vlm()
