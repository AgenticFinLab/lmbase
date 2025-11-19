"""
Interface of the MathVision dataset.
"""

import os
import ast
import re
import logging

from lmbase.identifier import MATH_SOLUTION_PROMPT
from lmbase.dataset.base import VisualTextSample, VisualTextBase


class MathVisionDataset(VisualTextBase):
    """A consistent interface for the MathVision dataset."""

    def __init__(
        self, split: str = "test", hf_dataname: str = None, config: dict = None
    ):
        self.data_path = config["data_path"]
        self.image_path = os.path.join(self.data_path, "images")
        os.makedirs(self.image_path, exist_ok=True)

        super().__init__(split=split, hf_dataname=hf_dataname, config=config)

    def to_format(self, sample: dict):
        """Get the sample from the given idx."""
        sample_id = sample["id"]
        # Create the sample
        question = sample["question"].strip()

        # extract all <imageN> tokens
        question_images = []
        image_tokens = re.findall(r"<image\d+>", question)
        for token in image_tokens:
            image_data = sample.get("decoded_image")
            if image_data is not None:
                filename = f"Image-ID{sample_id}-{token}"
                save_path = self.save_pil_image(image_data, self.image_path, filename)
                if save_path is not None:
                    question_images.append((token, save_path))
                else:
                    logging.warning(
                        "Failed to save image for %s in sample %s",
                        token,
                        sample_id,
                    )
            else:
                logging.warning("No decoded_image for sample %s", sample_id)

        # process the options
        options = sample.get("options", [])
        if options is None or len(options) == 0:
            question = f"{question} {MATH_SOLUTION_PROMPT}\n"
        else:
            try:
                if isinstance(options, str):
                    options = ast.literal_eval(options)
                option_letters = [chr(ord("A") + i) for i in range(len(options))]
                options_str = "\n".join(
                    [
                        f"({letter}): {opt}"
                        for letter, opt in zip(option_letters, options)
                    ]
                )
                question = f"{question} {MATH_SOLUTION_PROMPT}\nOptions:\n{options_str}"
            except Exception as e:
                logging.warning(
                    "Failed to parse options for sample %s: %s",
                    sample_id,
                    e,
                )

        cot_answer = sample.get("solution", "") or ""

        groundtruth = str(sample.get("answer", "")).strip()

        return VisualTextSample(
            main_id=sample_id,
            split=self.split,
            question=question,
            cot_answer=cot_answer,
            groundtruth=groundtruth,
            question_images=question_images,
            sample_info={
                "dataset": self.hf_dataname,
                "level": sample.get("level"),
                "subject": sample.get("subject"),
            },
        )
