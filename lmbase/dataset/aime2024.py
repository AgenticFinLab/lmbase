"""
Interface of the AIME2024 dataset.
"""

from fgreason.dataset.base import TextSample, VisualTextBase
from fgreason.identifier import MATH_SOLUTION_PROMPT


class AIME2024Dataset(VisualTextBase):
    """A consistent interface for the AIME2024 dataset."""

    def to_format(self, sample):
        """Get the sample from the given idx."""

        # Create the sample
        cot_answer = sample["solution"]
        # opt = re_utility.extract_format_equations(cot_answer,
        groundtruth_sol = sample["answer"]
        problem = sample["problem"]
        question = f"{problem} {MATH_SOLUTION_PROMPT}"

        return TextSample(
            main_id=sample["id"],
            split=self.split,
            question=question,
            cot_answer=cot_answer,
            groundtruth=groundtruth_sol,
            sample_info={
                "dataset": self.hf_dataname,
                "url": sample["url"],
                "year": sample["year"],
            },
        )
