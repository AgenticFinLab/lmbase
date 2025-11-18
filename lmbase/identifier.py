"""
The keyword used to identify the core solution of the output of LLMs.
"""

MATH_SOLUTION_FLAG = "\\boxed{...}"
OPTION_SOLUTION_FLAG = "\\boxed{...}"

MATH_SOLUTION_PROMPT = f"(Place final solution within {MATH_SOLUTION_FLAG})."
OPTION_SOLUTION_PROMPT = f"(Place final selected option within {OPTION_SOLUTION_FLAG})."
