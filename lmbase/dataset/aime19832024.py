"""
Interface of the AIME19832024 dataset.
"""

from lmbase.dataset.base import TextSample, VisualTextBase
from lmbase.identifier import MATH_SOLUTION_PROMPT


class AIME19832024Dataset(VisualTextBase):
    """A consistent interface for the AIME19832024 dataset."""

    def to_format(self, sample):
        """Get the sample from the given idx."""

        # Create the sample
        cot_answer = ""
        # opt = re_utility.extract_format_equations(cot_answer,
        groundtruth_sol = sample["Answer"]
        problem = sample["Question"]
        question = f"{problem} {MATH_SOLUTION_PROMPT}"

        return TextSample(
            main_id=sample["ID"],
            split=self.split,
            question=question,
            cot_answer=cot_answer,
            groundtruth=groundtruth_sol,
            sample_info={
                "dataset": self.hf_dataname,
                "year": sample["Year"],
                "problem_number": sample["Problem Number"],
            },
        )
