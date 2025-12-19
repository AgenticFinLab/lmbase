"""
Interface of the We-Math dataset.
"""

import os
import re
import logging

from lmbase.identifier import MATH_SOLUTION_PROMPT
from lmbase.dataset.base import VisualTextSample, VisualTextBase


class WeMathDataset(VisualTextBase):
    """A consistent interface for the We-Math dataset."""

    def __init__(
        self,
        split: str = "testmini",
        hf_dataname: str = None,
        config: dict = None,
    ):
        self.data_path = config["data_path"]
        self.image_path = os.path.join(self.data_path, "images")
        os.makedirs(self.image_path, exist_ok=True)

        super().__init__(split=split, hf_dataname=hf_dataname, config=config)

    def to_format(self, sample: dict):
        """Get the sample from the given idx."""
        # Create the sample
        sample_id = sample["question number"]

        question = sample["question"].strip()

        # extract all <imageN> tokens
        question_images = []
        image_tokens = re.findall(r"<image\d+>", question)

        # If there are no image tokens in the question but there is an image, add image tokens to the question
        image_data = sample.get("image_path")
        if image_data is not None and not image_tokens:
            # If image_data is a list or tuple, count the number of images
            if isinstance(image_data, (list, tuple)):
                num_images = len(image_data)
            else:
                # If it's a single image, count as 1
                num_images = 1

            # Add image tokens to the beginning of the question
            for i in range(num_images):
                image_token = f"<image{i+1}>"
                image_tokens.append(image_token)

            # Update the question with the new image tokens
            image_tokens_str = " ".join(image_tokens)
            question = image_tokens_str + " " + question

        for token in image_tokens:
            if image_data is not None:
                # Handle single image or multiple images
                if isinstance(image_data, (list, tuple)):
                    # If image_data is a list/tuple, get the corresponding image based on token index
                    token_index = int(token.replace("<image", "").replace(">", "")) - 1
                    if 0 <= token_index < len(image_data):
                        current_image = image_data[token_index]
                    else:
                        continue
                else:
                    # If it's a single image, use it for the first image token
                    token_index = int(token.replace("<image", "").replace(">", "")) - 1
                    # Only use single image for <image1>
                    if token_index == 0:
                        current_image = image_data
                    else:
                        logging.warning(
                            "More image tokens than available images in sample %s",
                            sample_id,
                        )
                        continue

                filename = f"Image-ID{sample_id}-{token}"
                save_path = self.save_pil_image(
                    current_image, self.image_path, filename
                )
                if save_path is not None:
                    question_images.append((token, save_path))
                else:
                    logging.warning(
                        "Failed to save image for %s in sample %s",
                        token,
                        sample_id,
                    )
            else:
                logging.warning("No image data for sample %s", sample_id)

        # process the options
        options = sample["option"]
        # if no options
        if options is None or len(options) == 0:
            question = f"{question} {MATH_SOLUTION_PROMPT}\n"
        # split the options
        else:
            try:
                if isinstance(options, str):
                    # Parse the string format "A. 3.14; B. 6.28; C. 12.56; D. Cannot be determined; E. No correct answer"
                    # Split by semicolon and extract option content
                    parts = [part.strip() for part in options.split(";")]
                    parsed_options = []

                    for part in parts:
                        # Remove leading letter and dot (e.g., "A. 3.14" -> "3.14")
                        match = re.match(r"^[A-Z]\.\s*(.*)", part.strip())
                        if match:
                            parsed_options.append(match.group(1).strip())
                        else:
                            # If doesn't match expected format, keep as is
                            parsed_options.append(part.strip())

                    options = parsed_options

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
                # Fallback to original string
                question = f"{question} {MATH_SOLUTION_PROMPT}\nOptions:\n{options}"

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
                "ID": sample.get("ID"),
                "knowledge concept": sample.get("knowledge concept"),
                "key": sample.get("key"),
                "knowledge concept description": sample.get(
                    "knowledge concept description"
                ),
            },
        )
