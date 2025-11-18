"""
Interface of the A-OKVQA dataset.
"""

import os
import ast
import re
import logging

from fgreason.identifier import OPTION_SOLUTION_PROMPT
from fgreason.dataset.base import VisualTextSample, VisualTextBase


class AOKVQADataset(VisualTextBase):
    """A consistent interface for the A-OKVQA dataset."""

    def __init__(
        self, split: str = "train", hf_dataname: str = None, config: dict = None
    ):
        self.data_path = config["data_path"]
        self.image_path = os.path.join(self.data_path, "images")
        os.makedirs(self.image_path, exist_ok=True)

        super().__init__(split=split, hf_dataname=hf_dataname, config=config)

    def to_format(self, sample: dict):
        """Get the sample from the given idx."""
        sample_id = sample["question_id"]
        # Create the sample
        question = sample["question"].strip()

        # Check the number of images and add corresponding <image> tags
        image_data = sample.get("image")
        question_images = []

        if image_data is not None:
            # Get the number of images (assuming image_data is a list or has length property)
            if isinstance(image_data, list):
                image_count = len(image_data)
            else:
                # If it's a single image object, count is 1
                image_count = 1
                # Wrap single image into a list for subsequent processing
                image_data = (
                    [image_data] if not isinstance(image_data, list) else image_data
                )

            # Add corresponding number of <image> tags before the question
            for i in range(image_count):
                question = f"<image{i+1}>{question}"

            # Save each image
            for i in range(image_count):
                if i < len(image_data):  # Ensure index doesn't exceed bounds
                    filename = f"Image-ID{sample_id}-image{i+1}"
                    save_path = self.save_pil_image(
                        image_data[i], self.image_path, filename
                    )
                    if save_path is not None:
                        question_images.append((f"<image{i+1}>", save_path))
                    else:
                        logging.warning(
                            f"Failed to save image{i+1} for sample {sample_id}"
                        )
                else:
                    logging.warning(
                        f"Image data missing for index {i} in sample {sample_id}"
                    )
        else:
            logging.warning(f"No image data for sample {sample_id}")

        # process the options
        options = sample.get("choices", [])
        if options is None or len(options) == 0:
            question = f"{question} {OPTION_SOLUTION_PROMPT}\n"
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
                question = (
                    f"{question} {OPTION_SOLUTION_PROMPT}\nOptions:\n{options_str}"
                )
            except Exception as e:
                logging.warning(f"Failed to parse options for sample {sample_id}: {e}")

        cot_answer = sample.get("rationales", "") or ""

        groundtruth_index = sample.get("correct_choice_idx", "") or ""
        options = sample.get("choices", [])

        if groundtruth_index == "":
            groundtruth = ""
        else:
            groundtruth = str(options[int(groundtruth_index)]).strip()

        return VisualTextSample(
            main_id=sample_id,
            split=self.split,
            question=question,
            cot_answer=cot_answer,
            groundtruth=groundtruth,
            question_images=question_images,
            sample_info={
                "dataset": self.hf_dataname,
                "direct_answers": sample.get("direct_answers"),
                "difficult_direct_answer": sample.get("difficult_direct_answer"),
            },
        )
