"""
Interface of the MultiArith dataset.

Dataset Source: https://huggingface.co/datasets/ChilleD/MultiArith

Description:
    MultiArith is a dataset of arithmetic word problems designed for evaluating
    chain-of-thought (CoT) reasoning in large language models. It contains multi-step
    math word problems with step-by-step rationales and final numeric answers.

Size: 600 rows (train: 420, test: 180)

Splits:
    - train: 420 examples
    - test: 180 examples

Features:
    - question: The math word problem statement
    - final_ans: The final numeric answer

License: Not specified (see dataset repository)
Language: English
Paper: Zero-shot CoT (Kojima et al., NeurIPS 2022)
"""

from lmbase.dataset.base import TextSample, VisualTextBase


class MultiArithDataset(VisualTextBase):
    """A consistent interface for the MultiArith dataset."""

    def to_format(self, sample):
        """Get the sample from the given idx."""

        # Create the sample
        self.idx += 1

        # Create the question
        question = sample["question"]
        question = f"{question}{self.SOLUTION_FORMAT_PROMPT}"

        # extract the groundtruth
        groundtruth = sample["final_ans"]

        return TextSample(
            main_id=f"ID{self.idx}",
            split=self.split,
            question=question,
            cot_answer=groundtruth,
            groundtruth=groundtruth,
            sample_info={
                "dataset": self.hf_dataname,
            },
        )
