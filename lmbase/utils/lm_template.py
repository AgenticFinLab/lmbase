"""
Process the chat template of large models.
"""


def get_template_parts(model_name: str):
    """Get different parts of the sample."""
    # As different llms use different prompt templates, we need to set the
    # instruction and response accordingly to avoid the issues of
    # 'NAN loss' and 'ZeroDivisionError: division by zero'
    instruction_flag = None
    response_flag = None
    if "Llama-3.2" in model_name:
        instruction_flag = "<|start_header_id|>user<|end_header_id|>\n\n"
        response_flag = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    if "Qwen2.5" in model_name:
        instruction_flag = "<|im_start|>user\n"
        response_flag = "<|im_start|>assistant\n"

    return instruction_flag, response_flag
