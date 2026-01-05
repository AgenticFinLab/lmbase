"""
Interface of the GPQA-Diamond dataset.
"""

from lmbase.dataset.base import TextSample, VisualTextBase


class GPQADiamondDataset(VisualTextBase):
    """A consistent interface for the GPQA-Diamond dataset."""

    def to_format(self, sample):
        """Get the sample from the given idx."""
        self.idx += 1

        # Create the sample
        # The question field in GPQA-Diamond usually contains the question and options
        question = sample["question"]
        question = f"{question}{self.SOLUTION_FORMAT_PROMPT}"

        # The answer is the correct option letter (e.g., "A", "B", "C", "D")
        groundtruth = sample["answer"]

        return TextSample(
            main_id=f"ID{self.idx}",
            split=self.split,
            question=question,
            cot_answer="",  # GPQA-Diamond typically doesn't provide CoT in the main subset
            groundtruth=groundtruth,
            sample_info={
                "dataset": self.hf_dataname,
            },
        )
