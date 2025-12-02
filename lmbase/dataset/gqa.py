"""
Interface of the GQA dataset.

This implementation follows the VisualTextBase pattern and formats samples
for vision-language model evaluation. It saves associated images and constructs
question text with explicit image tokens when present.
"""

import os
import logging
import re

from datasets import load_dataset

from lmbase.identifier import FINAL_SOLUTION_FLAG
from lmbase.dataset.base import VisualTextSample, VisualTextBase


class GQADataset(VisualTextBase):
    """A consistent interface for the GQA dataset."""

    def __init__(
        self,
        split: str = "train",
        hf_dataname: str = None,
        config: dict = None,
    ):
        # Prepare paths for saving images
        self.data_path = config["data_path"]
        self.image_path = os.path.join(self.data_path, "images")
        os.makedirs(self.image_path, exist_ok=True)

        # Images of the dataset
        self.hf_images = None

        super().__init__(
            split=split,
            hf_dataname=hf_dataname,
            config=config,
        )

    def map_dataset(self):
        """Map the dataset to the desired format."""

        if self.hf_dataset is None:
            # Download the images to the local disk
            self.hf_images = load_dataset(
                self.hf_dataname,
                f"{self.split}_all_images",
            )
            self.hf_dataset = load_dataset(
                self.hf_dataname,
                f"{self.split}_all_instructions",
            )
            # Images are accessed by id directly from `self.hf_images` in `to_format`
        logging.info(
            "   - Mapping samples to lmbase format, i.e., lmbase.dataset.base.TextSample"
        )
        # Make the sample to be the desired format defined
        # in the dataset.base class
        self.hf_dataset = self.hf_dataset.map(
            self.batch_format,
            batched=True,
            batch_size=1000,
            load_from_cache_file=True,
            remove_columns=self.hf_dataset.column_names,
        )

        # Save some demo samples to the dataset folder
        self.save_example_samples(num_samples=20)

    def _image_by_id(self, image_id):
        rows = self.hf_images.filter(lambda x: x["id"] == image_id)
        return rows[0]["image"] if len(rows) > 0 else None

    def to_format(self, sample: dict):
        """Get the sample from the given idx."""
        sample_id = sample["id"]

        question = str(sample["question"]).strip()

        question_images = []
        image_tokens = re.findall(r"<image\d+>", question)
        image_id = sample["imageId"]
        image_data = self._image_by_id(image_id)

        if image_tokens:
            for token in image_tokens:
                if image_data is not None:
                    filename = f"Image-ID{image_id}-{token}"
                    save_path = self.save_pil_image(
                        image_data, self.image_path, filename
                    )
                    if save_path is not None:
                        question_images.append((token, save_path))
        else:
            if image_data is not None:
                question = f"<image1>{question}"
                token = "<image1>"
                filename = f"Image-ID{image_id}-{token}"
                save_path = self.save_pil_image(image_data, self.image_path, filename)
                if save_path is not None:
                    question_images.append((token, save_path))

        question = f"{question} {FINAL_SOLUTION_FLAG}\n"

        cot_answer = str(sample["fullAnswer"]).strip()
        groundtruth = str(sample["answer"]).strip()

        return VisualTextSample(
            main_id=sample_id,
            split=self.split,
            question=question,
            cot_answer=cot_answer,
            groundtruth=groundtruth,
            question_images=question_images,
            sample_info={
                "dataset": self.hf_dataname,
                "isBalanced": sample["isBalanced"],
                "groups": sample["groups"],
                "entailed": sample["entailed"],
                "equivalent": sample["equivalent"],
                "types": sample["types"],
                "annotations": sample["annotations"],
                "semantic": sample["semantic"],
                "semanticStr": sample["semanticStr"],
            },
        )
