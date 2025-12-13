"""
Interface of the DAPO-Math-17k dataset.
"""

import re

from lmbase.dataset.base import TextSample, VisualTextBase
from lmbase.identifier import MATH_SOLUTION_PROMPT


class DAPOMathDataset(VisualTextBase):
    """A consistent interface for the DAPO-Math-17k dataset."""

    def to_format(self, sample):
        """Get the sample from the given idx."""

        # Create the sample
        id = sample["extra_info"]["index"]

        # extract the groundtruth
        groundtruth = sample["reward_model"]["ground_truth"]

        # Extract content from the first item
        content = sample["prompt"][0]["content"]

        # The question is wrapped in the content, for example:
        # "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\nLet $a, b, c$ be distinct numbers such that the equations $x^2 + ax + 1 = 0$ and $x^2 + bx + c = 0$ have a common real root, and the equations $x^2 + x + a = 0$ and $x^2 + cx + b = 0$ also have a common real root. Compute the sum $a + b + c$.\n\nRemember to put your answer on its own line after \"Answer:\".\n\n"
        # Use regex to extract the text between the two specified phrases to get the question
        pattern = r"to the problem\.\n\n([\s\S]*?)\n\nRemember"
        match = re.search(pattern, content)
        question = match.group(1).strip()

        # Create the question
        question = f"{question} {MATH_SOLUTION_PROMPT}"

        return TextSample(
            main_id=id,
            split=self.split,
            question=question,
            cot_answer="",
            groundtruth=groundtruth,
            sample_info={
                "dataset": self.hf_dataname,
                "ability": sample["ability"],
                "data_source": sample["data_source"],
                "reward_model_style": sample["reward_model"]["style"],
            },
        )
