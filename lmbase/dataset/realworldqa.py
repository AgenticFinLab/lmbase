"""
Interface of the RealWorldQA dataset.

Dataset Source: https://huggingface.co/datasets/xai-org/RealworldQA

Description:
    A benchmark designed for real-world understanding. The images are
    anonymized photographs taken from vehicles, plus other real-world images,
    each paired with one question and an easily verifiable answer. Questions are
    mostly open-ended short answers, while some samples are multiple choice with
    the options embedded in the question.

Size: ~765 rows

Splits:
    - test: Evaluation set (~765 examples)

Features:
    - image: Associated image
    - question: The question text, which may already contain the answering
      instruction and the options, e.g., "A. Yes\nB. No\nPlease answer directly
      with only the letter of the correct option and nothing else."
    - answer: Verifiable answer, i.e., the option letter for multiple choice
      samples or the short answer text otherwise

License: CC BY-ND 4.0
Language: English
"""

import os
import logging

from lmbase.dataset.base import VisualTextSample, VisualTextBase


class RealWorldQADataset(VisualTextBase):
    """A consistent interface for the RealWorldQA dataset."""

    def __init__(
        self, split: str = "test", hf_dataname: str = None, config: dict = None
    ):
        self.data_path = config["data_path"]
        self.image_path = os.path.join(self.data_path, "images")
        os.makedirs(self.image_path, exist_ok=True)

        super().__init__(split=split, hf_dataname=hf_dataname, config=config)

    def to_format(self, sample: dict):
        """Get the sample from the given idx."""
        # The dataset does not provide an ID, use the visiting index instead
        self.idx += 1
        sample_id = f"ID{self.idx}"

        question = str(sample["question"]).strip()

        question_images = []
        image_token = "<image1>"
        image_data = sample.get("image")
        if image_data is not None:
            question = f"{image_token}{question}"
            filename = f"RealWorldQA-{sample_id}-image1"
            save_path = self.save_pil_image(image_data, self.image_path, filename)
            if save_path is not None:
                question_images.append((image_token, save_path))
            else:
                logging.warning("Failed to save image for sample %s", sample_id)
        else:
            logging.warning("No image data for sample %s", sample_id)

        # The questions already carry the answering instruction of the dataset,
        # e.g., "answer directly with only the letter", so only the configurable
        # solution prompt is appended (empty by default)
        question = f"{question}{self.SOLUTION_FORMAT_PROMPT}"

        answer = str(sample["answer"]).strip()

        return VisualTextSample(
            main_id=sample_id,
            split=self.split,
            question=question,
            cot_answer=answer,
            groundtruth=answer,
            question_images=question_images,
            sample_info={
                "dataset": self.hf_dataname,
            },
        )
