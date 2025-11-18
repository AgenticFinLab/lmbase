"""
Interface of the CodeAlpaca dataset.
"""

from lmbase.dataset.base import TextCodeSample, VisualTextBase


class CodeAlpacaDataset(VisualTextBase):
    """A consistent interface for the CodeAlpaca dataset."""

    def to_format(self, sample):
        """Get the sample from the given idx."""
        self.idx += 1

        # Create the sample
        problem = sample["instruction"]
        question = f"{problem}"
        return TextCodeSample(
            main_id=f"ID{self.idx}",
            split=self.split,
            question=question,
            cot_answer="",
            groundtruth=sample["output"],
            sample_info={"dataset": self.hf_dataname},
        )
