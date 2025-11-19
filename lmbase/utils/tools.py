"""
Tools used by the whole project.
"""

import re
from typing import List


def format_term(terminology: str):
    """Format the terminology to be the standard one.
    This function ensure that all the terminology are in the same format.
    """
    # basic conversion
    terminology = (
        terminology.replace("_", " ").replace("and", "&").replace("-", " ").rstrip()
    )
    if terminology.isupper():
        return terminology

    return terminology.title()


def remove_step_identifiers(decomposed_text):
    """
    Remove 'Step X:' identifiers from decomposed steps.
    """
    return re.sub(r"Step\s*\d+\s*:\s*", "", decomposed_text)


def normalize_text(text):
    """
    Normalize text by collapsing all whitespace (spaces, tabs, newlines) into a single space.
    """
    text = re.sub(
        r"\s+", " ", text
    )  # Replace any whitespace sequence with a single space
    return text.strip()  # Remove leading/trailing spaces


def check_match(original_answers, decomposed_steps_list):
    """
    Check if each original answer matches the corresponding decomposed steps after removing identifiers.

    Args:
        original_answers (list of str): List of original answers.
        decomposed_steps_list (list of str): List of decomposed answers with step identifiers.

    Returns:
        list of bool: List indicating match (True) or mismatch (False) for each pair.
    """
    results = []
    for original, decomposed in zip(original_answers, decomposed_steps_list):
        cleaned_decomposed = remove_step_identifiers(decomposed)
        original_normalized = normalize_text(original)
        decomposed_normalized = normalize_text(cleaned_decomposed)

        results.append(original_normalized == decomposed_normalized)
    return results


def get_step_content(batch_steps: List[str]):
    """
    Extract the content of each step from the whole string.

    Each item of `batch_steps` is a string holding:
        "Step idx: ...."
    This function is to extract the '...' while ignoring the 'Step idx:' part.
    """
    return [
        re.findall(
            r"((?:Step|step)\s*\d+\s*:\s*.*?)(?=(Step\s*\d+\s*:|step\s*\d+\s*:|$))",
            steps,
            re.DOTALL,
        )
        for steps in batch_steps
    ]
