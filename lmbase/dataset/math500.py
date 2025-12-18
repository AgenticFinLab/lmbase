"""
Interface of the MATH-500 dataset.
"""

from lmbase.dataset.base import TextSample, VisualTextBase
from lmbase.identifier import MATH_SOLUTION_PROMPT


class Math500Dataset(VisualTextBase):
    """A consistent interface for the MATH-500 dataset."""

    def to_format(self, sample):
        """Get the sample from the given idx."""

        # Create the sample
        self.idx += 1

        # Create the question
        question = sample["problem"]
        question = f"{question} {MATH_SOLUTION_PROMPT}"

        # extract the groundtruth
        groundtruth = sample["answer"]

        # extract the cot_answer
        cot_answer = sample["solution"]

        return TextSample(
            main_id=f"ID{self.idx}",
            split=self.split,
            question=question,
            cot_answer=cot_answer,
            groundtruth=groundtruth,
            sample_info={
                "dataset": self.hf_dataname,
                "level": sample["level"],
                "subject": sample["subject"],
                "unique_id": sample["unique_id"],
            },
        )
