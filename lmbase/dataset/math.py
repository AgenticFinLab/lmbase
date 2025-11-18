"""
Interface of the MATH dataset.
"""

from math_verify import LatexExtractionConfig, parse

from fgreason.dataset.base import TextSample, VisualTextBase
from fgreason.identifier import MATH_SOLUTION_PROMPT


class MATHDataset(VisualTextBase):
    """A consistent interface for the MATH dataset."""

    def to_format(self, sample):
        """Get the sample from the given idx."""
        self.idx += 1

        # Create the sample
        cot_answer = sample["solution"]
        # opt = re_utility.extract_format_equations(cot_answer, target_format="\\boxed")
        # groundtruth_sol = "" if opt is None else opt[-1]
        # The parsed item will be a list holding a value and a str value
        groundtruth_sol = parse(
            cot_answer,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        groundtruth_sol = "" if len(groundtruth_sol) == 0 else groundtruth_sol[-1]
        problem = sample["problem"]
        question = f"{problem} {MATH_SOLUTION_PROMPT}"

        return TextSample(
            main_id=f"ID{self.idx}",
            split=self.split,
            question=question,
            cot_answer=cot_answer,
            groundtruth=groundtruth_sol,
            sample_info={
                "dataset": self.hf_dataname,
                "level": sample["level"],
                "type": sample["type"],
            },
        )
