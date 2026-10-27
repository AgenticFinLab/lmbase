"""
Interface of the MMStar dataset.

Dataset Source: https://huggingface.co/datasets/Lin-Chen/MMStar

Description:
    A challenging multimodal benchmark that evaluates whether large
    vision-language models truly need visual information. All 1,500 samples are
    carefully selected to be visually indispensable, and they are balanced
    across 6 core capabilities (coarse perception, fine-grained perception,
    instance reasoning, logical reasoning, math, science & technology) and 18
    detailed capability categories.

Size: ~1,500 rows

Configurations:
    - val: The only configuration released by the dataset
    Config setting in code: config name "val"

Splits:
    - val: Evaluation set (~1,500 examples)

Features:
    - index: Unique identifier of the sample
    - question: The question text with the options embedded, e.g.,
      "Which option ... ?\nOptions: A: ..., B: ..., C: ..., D: ..."
    - image: Associated image
    - answer: Correct option letter, e.g., "A"
    - category: Core capability, e.g., "coarse perception"
    - l2_category: Fine-grained capability, e.g., "image scene and topic"
    - meta_info: Source metadata, i.e., {"source", "split", "image_path"}

License: Not specified (see dataset repository)
Language: English
Paper: arXiv:2403.20330
"""

import os
import logging

from datasets import load_dataset

from lmbase.identifier import OPTION_SOLUTION_PROMPT
from lmbase.dataset.base import VisualTextSample, VisualTextBase


class MMStarDataset(VisualTextBase):
    """A consistent interface for the MMStar dataset."""

    def __init__(
        self, split: str = "val", hf_dataname: str = None, config: dict = None
    ):
        self.data_path = config["data_path"]
        self.image_path = os.path.join(self.data_path, "images")
        os.makedirs(self.image_path, exist_ok=True)

        super().__init__(split=split, hf_dataname=hf_dataname, config=config)

    def map_dataset(self):
        """Map the dataset to the desired format."""

        if self.hf_dataset is None:
            # MMStar only releases the `val` configuration and split
            self.hf_dataset = load_dataset(self.hf_dataname, "val", split=self.split)
        super().map_dataset()

    def to_format(self, sample: dict):
        """Get the sample from the given idx."""
        sample_id = sample["index"]

        question = str(sample["question"]).strip()

        question_images = []
        image_token = "<image1>"
        image_data = sample.get("image")
        if image_data is not None:
            question = f"{image_token}{question}"
            filename = f"MMStar-ID{sample_id}-image1"
            save_path = self.save_pil_image(image_data, self.image_path, filename)
            if save_path is not None:
                question_images.append((image_token, save_path))
            else:
                logging.warning("Failed to save image for sample %s", sample_id)
        else:
            logging.warning("No image data for sample %s", sample_id)

        # The options are embedded in the question, only the solution prompt
        # is appended so that the final option can be extracted downstream
        question = f"{question} {OPTION_SOLUTION_PROMPT}"

        meta_info = sample.get("meta_info") or {}

        return VisualTextSample(
            main_id=sample_id,
            split=self.split,
            question=question,
            cot_answer="",
            groundtruth=str(sample["answer"]).strip(),
            question_images=question_images,
            sample_info={
                "dataset": self.hf_dataname,
                "category": sample["category"],
                "l2_category": sample["l2_category"],
                "source": meta_info.get("source"),
                "image_path": meta_info.get("image_path"),
            },
        )
