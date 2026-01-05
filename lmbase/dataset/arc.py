"""
Interface of the ARC dataset.
"""

from datasets import load_dataset

from lmbase.dataset.base import TextSample, VisualTextBase


class ARCDataset(VisualTextBase):
    """A consistent interface for the ARC dataset."""

    def map_dataset(self):
        """Map the dataset to the desired format."""

        # ARC has two subsets: ARC-Challenge and ARC-Easy
        # We default to ARC-Challenge if not specified in config, but since registry
        # doesn't pass subset name easily, we might need to handle it.
        # However, the user said "only two subsets". Usually users mean ARC-Challenge.
        # Let's load ARC-Challenge by default as it's the standard benchmark.
        # If we need to support both, we might need a way to specify it.
        # For now, let's assume ARC-Challenge as it's the primary one.

        subset = self.config["subset"]

        self.hf_dataset = load_dataset(
            self.hf_dataname,
            subset,
            split=self.split,
        )

        super().map_dataset()

    def to_format(self, sample: dict):
        """Get the sample from the given idx."""

        # ARC dataset has 'id', 'question', 'choices', 'answerKey'
        # choices is a struct with 'text' (list of strings) and 'label' (list of strings like A, B, C, D)

        # Create the sample
        question_text = sample["question"]

        choices = sample["choices"]
        texts = choices["text"]
        labels = choices["label"]

        # Format options
        options_str = ""
        if texts and labels:
            option_lines = [f"({label}) {text}" for label, text in zip(labels, texts)]
            options_str = "\n".join(option_lines)

        if options_str:
            question = f"{question_text}\n{options_str}{self.SOLUTION_FORMAT_PROMPT}"
        else:
            question = f"{question_text}{self.SOLUTION_FORMAT_PROMPT}"

        groundtruth = sample["answerKey"]

        return TextSample(
            main_id=sample["id"],
            split=self.split,
            question=question,
            cot_answer="",
            groundtruth=groundtruth,
            sample_info={
                "dataset": self.hf_dataname,
                "subset": self.config["subset"],
            },
        )
