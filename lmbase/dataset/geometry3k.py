"""
Interface of the geometry3k dataset.
"""

import os
import re
import logging
from lmbase.dataset.base import VisualTextSample, VisualTextBase


class Geometry3kDataset(VisualTextBase):
    """A consistent interface for the geometry3k dataset."""

    def __init__(
        self,
        split: str = "train",
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
        sample_id = self.idx
        self.idx += 1

        question = sample["problem"].strip()

        # extract all <imageN> tokens
        question_images = []
        image_tokens = re.findall(r"<image\d*>", question)

        # If there are no image tokens in the question but there is an image, add image tokens to the question
        image_data = sample.get("images")
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
            if image_data is None:
                logging.warning("No image data for sample %s", sample_id)
                continue

            # parse token index, default to 0 if no number
            num_str = token.replace("<image", "").replace(">", "")
            token_index = 0 if num_str == "" else int(num_str) - 1

            # handle multiple images
            if isinstance(image_data, (list, tuple)):
                if 0 <= token_index < len(image_data):
                    current_image = image_data[token_index]
                else:
                    logging.warning(
                        "Token %s index out of range in sample %s", token, sample_id
                    )
                    continue
            # handle single image
            else:
                if token_index == 0:
                    current_image = image_data
                else:
                    logging.warning(
                        "More image tokens than available images in sample %s",
                        sample_id,
                    )
                    continue

            filename = f"Image-ID{sample_id}-{token}"
            save_path = self.save_pil_image(current_image, self.image_path, filename)
            if save_path is not None:
                question_images.append((token, save_path))

        question = f"{question}{self.SOLUTION_FORMAT_PROMPT}"

        cot_answer = ""

        groundtruth = sample["answer"].strip()

        return VisualTextSample(
            main_id=sample_id,
            split=self.split,
            question=question,
            cot_answer=cot_answer,
            groundtruth=groundtruth,
            question_images=question_images,
            sample_info={
                "dataset": self.hf_dataname,
            },
        )
