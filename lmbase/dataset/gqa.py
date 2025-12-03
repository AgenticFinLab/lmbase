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

        # Cache for image lookup to avoid repeated filtering
        self._image_cache = None

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
            # Build image cache before processing samples to avoid repeated filtering
            self._build_image_cache()
            # Images are accessed by id directly from `self.hf_images` in `to_format`
        logging.info(
            "   - Mapping samples to lmbase format, i.e., lmbase.dataset.base.TextSample"
        )
        # Make the sample to be the desired format defined in the dataset.base class
        # Check if hf_dataset is a DatasetDict (contains multiple splits)
        # This is a DatasetDict
        if hasattr(self.hf_dataset, "keys"):
            # Process only the current split
            if self.split in self.hf_dataset:
                split_dataset = self.hf_dataset[self.split]
                column_names = split_dataset.column_names

                # Apply the mapping function to the specific split and replace the entire hf_dataset
                self.hf_dataset = split_dataset.map(
                    self.batch_format,
                    batched=True,
                    batch_size=1000,
                    load_from_cache_file=True,
                    # Remove all original columns
                    remove_columns=column_names,
                )
            else:
                raise ValueError(
                    f"Split '{self.split}' not found in dataset. Available splits: {list(self.hf_dataset.keys())}"
                )
        # This is a single Dataset
        else:
            # Get the column names for this dataset
            column_names = self.hf_dataset.column_names

            # Apply the mapping function
            self.hf_dataset = self.hf_dataset.map(
                self.batch_format,
                batched=True,
                batch_size=1000,
                load_from_cache_file=True,
                # Remove all original columns
                remove_columns=column_names,
            )
        # Save some demo samples to the dataset folder
        self.save_example_samples(num_samples=20)

    def _build_image_cache(self):
        """Build image cache to avoid repeated filtering operations."""
        if self._image_cache is not None:
            return  # Cache already built

        if isinstance(self.hf_images, dict) or hasattr(self.hf_images, "keys"):
            # If it's DatasetDict, get the dataset from the specified split
            available_splits = list(self.hf_images.keys())
            if available_splits:
                dataset = self.hf_images[self.split]
            else:
                self._image_cache = {}
                return
        else:
            # If it's already a single Dataset
            dataset = self.hf_images

        # Build mapping from image ID to image data
        self._image_cache = {}
        for item in dataset:
            self._image_cache[item["id"]] = item["image"]

    def _image_by_id(self, image_id):
        """Get image by ID using cached mapping instead of repeated filtering."""
        if self._image_cache is None:
            self._build_image_cache()

        return self._image_cache.get(image_id)

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
