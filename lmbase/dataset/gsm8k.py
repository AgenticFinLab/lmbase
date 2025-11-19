"""
Interface of the GSM8K dataset.

The "main" set of the GSM8K dataset is downloaded by default. These the training, validation, and test splits.
"""

from datasets import load_dataset

from lmbase.utils import re_extractor
from lmbase.identifier import MATH_SOLUTION_PROMPT
from lmbase.dataset.base import TextSample, VisualTextBase


class GSM8KDataset(VisualTextBase):
    """A consistent interface for the GSM8k dataset."""

    def map_dataset(self):
        """Map the dataset to the desired format."""

        self.hf_dataset = load_dataset(self.hf_dataname, "main", split=self.split)

        super().map_dataset()

    def to_format(self, sample):
        """Get the sample from the given idx."""
        self.idx += 1

        # Create the sample
        groundtruth_sol = re_extractor.extract_content(sample["answer"], marker="####")
        groundtruth_sol = "" if groundtruth_sol is None else groundtruth_sol
        problem = sample["question"]
        question = f"{problem} {MATH_SOLUTION_PROMPT}"
        return TextSample(
            main_id=f"ID{self.idx}",
            split=self.split,
            question=question,
            cot_answer=sample["answer"],
            groundtruth=groundtruth_sol,
            sample_info={"dataset": self.hf_dataname},
        )
