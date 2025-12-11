import os
from lmbase.inference.base import InferInput
from lmbase.inference.model_call import Qwen25VLInference, LLMInference


def demo_llm():
    lm_path = os.getenv("LLM_PATH", "Qwen/Qwen2.5-7B-Instruct")
    infer = LLMInference(lm_path=lm_path, inference_config={"use_vllm": False})
    inp = InferInput(system_msg="", user_msg="Write a short greeting.")
    try:
        out = infer.run(inp, max_new_tokens=16, temperature=0.7)
        print("LLM response:", out.response)
    except Exception as e:
        print("LLM demo failed:", e)


def demo_vlm():
    vlm_path = os.getenv("VLM_PATH", "Qwen/Qwen2.5-VL-7B-Instruct")
    infer = Qwen25VLInference(lm_path=vlm_path)
    # Qwen-style content list; image items can be added when you have a valid image path
    content = [{"type": "text", "text": "Describe the provided content briefly."}]
    inp = InferInput(system_msg="", user_msg=content)
    try:
        # Pass generation parameters via constructor generation_config
        infer.generation_config.update({"max_new_tokens": 16, "temperature": 0.7})
        out = infer.run(inp)
        print("VLM response:", out.response)
    except Exception as e:
        print("VLM demo failed:", e)


if __name__ == "__main__":
    demo_llm()
    demo_vlm()
