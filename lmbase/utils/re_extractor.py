"""
A collection of utility functions supported by regular expressions.
"""

import re
from typing import List

import regex


def is_flag_string(text_str: str, flags: List[str]):
    """Check whether the text_str contains the flags."""
    return any([flag.lower() in text_str.lower() for flag in flags])


def extract_sentences(text_str: str, split_pattern=r"(?<!\d)\.\s+|\n"):
    """Extract the final sentence from the string."""
    # Set the split pattern of the regular expression
    pattern = split_pattern

    # Find all sentences in the paragraph
    sentences = re.split(pattern, text_str.rstrip())

    sentences = [sent for sent in sentences if not sent.isspace() and len(sent) != 0]

    # Extract the final sentence
    return sentences


def extract_flagged_conclusion(
    text_str: str, flags: List[str] = None, weights: List[int] = None
):
    """
    Extract the conclusion containing the flags.

    This function will count the number of flags * weights in each sentence and return the
    one 1) containing the most important flags and 2) the final one when there are multiple
    """
    sentences = extract_sentences(text_str)
    # Count each sentence matches how many flags
    sentence_matched = []
    for sent in sentences:
        sentence_matched.append(
            sum([int(flag in sent) * weights[idx] for idx, flag in enumerate(flags)])
        )

    # Get the sentence with the most flags
    n_matched = len(sentence_matched)
    max_matched = max(sentence_matched)

    # Reverse the list in order to get the index of the final max values
    sentence_matched.reverse()
    # Here -1 as the index starts from 0
    index = n_matched - 1 - sentence_matched.index(max_matched)

    return sentences[index]


def extract_figures(
    text_str: str,
    paired_format="$",
):
    """Extract the figure results from the paired_format."""
    # This pattern is used to extract the target result from the given
    # `target_format`
    # For example, when target_format is $
    # This pattern can extract
    # $6$, $14$ -> 6, 14
    # $6.5$, $14.88$ -> 6.5, 14.88
    # $6, 7$ -> 6, 7
    # $6.5, 6.7$ -> 6.5, 6.7
    # $7.000.222$, $1000,00,0$ -> 7.000.222, 1000,00,0

    pattern = rf"\{paired_format}?(\d[\d,.]*(?:\.\d*)?)\{paired_format}?,?"

    # Find all matches in the text
    matches = re.findall(pattern, text_str)

    if not matches:
        return None

    return matches


def extract_content(text, marker, content_pattern=r"(\d+)"):
    """
    Extracts content, specifically the digital numbers, presented
    after the `marker` in a text.

    For example:
        when marker is #### and the text is "The answer is #### 1234",
        we will have 1234.
    """
    # Build the regex pattern by escaping the marker so it is taken literally.
    # Then, we allow optional whitespace and capture the desired content.
    pattern = re.escape(marker) + r"\s*" + content_pattern
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None


def extract_equations(text_str: str, target_format="$"):
    """Extract the target equation from the text_str."""
    target_format = "$"
    # Define a regular expression pattern to match the desired substrings
    pattern = rf"\{target_format}+(.*?)\{target_format}+"

    # Use re.findall() to find all matches
    matches = re.findall(pattern, text_str)
    # Extract the matched numbers
    numbers = [match for match in matches if match]

    # Once nothing is extract, just return the original text_str
    if not numbers:
        numbers = [text_str]

    return numbers


def extract_format_equations(
    text_str: str, equation_format="=", target_format="\\boxed"
):
    """
    Extract the equations in the format defined by `target_format` from
    the content after the equation_format.
    """
    # First extract the equation
    splitted_eq = text_str.split(equation_format)
    right_eq = splitted_eq[-1]
    escaped_marker = re.escape(target_format)
    pattern_str = rf"{escaped_marker}\{{(?P<content>(?:[^{{}}]|\{{(?&content)\}})*)\}}"
    pattern = regex.compile(pattern_str)
    # Extract the target result within the target_format
    matches = pattern.findall(right_eq)
    # pattern = rf"{re.escape(target_format)}{{((?:[^{{}}]+|{{[^{{}}]+}})+)}}"
    # matches = re.findall(pattern, right_eq)
    if not matches:
        return None
    return matches
