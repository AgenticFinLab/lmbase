"""
Interface of the hfcodealpaca dataset.
"""

from lmbase.dataset.base import TextCodeSample, VisualTextBase


class CodeAlpacaDataset(VisualTextBase):
    """A consistent interface for the hfCodeAlpaca dataset."""

    def to_format(self, sample):
        """Get the sample from the given idx."""
        self.idx += 1

        # Create the sample; handle both instruction/output and prompt/completion schemas.
        if "instruction" in sample:
            problem = sample["instruction"]
            groundtruth = sample.get("output", "")
        else:
            problem = sample.get("prompt", "")
            groundtruth = sample.get("completion", "")
        question = f"{problem}"
        return TextCodeSample(
            main_id=f"ID{self.idx}",
            split=self.split,
            question=question,
            cot_answer="",
            groundtruth=groundtruth,
            sample_info={"dataset": self.hf_dataname},
        )
