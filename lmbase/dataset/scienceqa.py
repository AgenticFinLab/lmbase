"""
Interface of the ScienceQA dataset.
"""

import os

from datasets import load_dataset

from lmbase.dataset.base import VisualTextSample, VisualTextBase
from lmbase.identifier import OPTION_SOLUTION_PROMPT


class ScienceQADataset(VisualTextBase):
    """A consistent interface for the ScienceQA dataset."""

    def __init__(
        self, split: str = "train", hf_dataname: str = None, config: dict = None
    ):

        self.data_path = config["data_path"]
        self.image_path = f"{self.data_path}/images"
        os.makedirs(self.image_path, exist_ok=True)

        super().__init__(split=split, hf_dataname=hf_dataname, config=config)

    def map_dataset(self):
        """Map the dataset to the desired format."""

        self.hf_dataset = load_dataset(
            self.hf_dataname, "ScienceQA-FULL", split=self.split
        )

        super().map_dataset()

    def to_format(self, sample: dict):
        """Get the sample from the given idx."""
        self.idx += 1

        # Create the sample
        question = sample["question"]
        question = f"{question} {OPTION_SOLUTION_PROMPT}."
        options = sample["choices"]
        image_data = sample["image"]
        q_image = None

        filename = f"{self.split}-Image-ID{self.idx}"
        filepath = f"{self.image_path}/{filename}.jpg"
        if os.path.exists(filepath):
            q_image = filepath
        else:
            save_path = self.save_pil_image(image_data, self.image_path, filename)
            if save_path is not None:
                q_image = save_path

        if options is None or len(options) == 0:
            question = f"{question}\n"
        else:
            option_letters = [chr(ord("A") + num) for num in range(len(options))]
            options_str = [
                f"({letter}): {choice}"
                for choice, letter in zip(options, option_letters)
            ]
            options_str = "\n".join(options_str)

            question = f"{question}\nOptions:\n{options_str}"

        groundtruth = chr(ord("A") + int(sample["answer"]))
        lecture = sample["lecture"]
        solution = sample["solution"]
        cot_answer = f"{lecture}\n{solution}"

        return VisualTextSample(
            main_id=f"ID{self.idx}",
            split=self.split,
            question=question,
            cot_answer=cot_answer,
            groundtruth=groundtruth,
            question_images=[("image", q_image)],
            sample_info={
                "dataset": self.hf_dataname,
                "grade": sample["grade"],
                "subject": sample["subject"],
                "topic": sample["topic"],
                "category": sample["category"],
                "skill": sample["skill"],
                "lecture": sample["lecture"],
                "hint": sample["hint"],
            },
        )
