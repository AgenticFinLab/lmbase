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
        self.image_path_by_id = {}

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
            for row in self.hf_images:
                img_id = row["id"]
                img = row["image"]
                filename = f"{self.split}-Image-ID{img_id}"
                save_path = self.save_pil_image(img, self.image_path, filename)
                if save_path is not None:
                    self.image_path_by_id[img_id] = save_path
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

    def to_format(self, sample: dict):
        """Get the sample from the given idx."""
        sample_id = sample["id"]

        question = str(sample["question"]).strip()

        question_images = []
        image_tokens = re.findall(r"<image\d+>", question)
        image_id = sample["imageId"]
        image_path = self.image_path_by_id[image_id]

        if image_tokens:
            for token in image_tokens:
                if image_path is not None:
                    question_images.append((token, image_path))
                elif image_id is not None and len(str(image_id)) > 0:
                    placeholder_path = os.path.join(self.image_path, f"{image_id}.jpg")
                    question_images.append((token, placeholder_path))
        else:
            if image_path is not None or (
                image_id is not None and len(str(image_id)) > 0
            ):
                question = f"<image1>{question}"
                token = "<image1>"
                if image_path is not None:
                    question_images.append((token, image_path))
                else:
                    placeholder_path = os.path.join(self.image_path, f"{image_id}.jpg")
                    question_images.append((token, placeholder_path))

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
