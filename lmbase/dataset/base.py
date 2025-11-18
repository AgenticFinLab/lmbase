"""
Interface of the base dataset.
"""

import os
import json
import random
import logging
from typing import List, Tuple
from dataclasses import dataclass

from datasets import load_dataset
from torch.utils.data import Dataset
from transformers.utils import ModelOutput as FieldFrozenContainer


@dataclass
class TextSample(FieldFrozenContainer):
    """
    The sample of the unimodal dataset.
    """

    # ID of the sample
    # By default, it is the index of the sample in the dataset
    # Otherwise, it can be the ID of the sample in the dataset
    main_id: str = None
    # Split:
    split: str = None
    # The question to be answered
    question: str = None
    # Answer in the chain of thought format
    cot_answer: str = None
    # groundtruth solution
    groundtruth: str = None
    # parse_groundtruth: str = None
    # Additional field to hold the information
    # of this sample
    sample_info: dict = None


@dataclass
class TextCodeSample(TextSample):
    """
    The coding sample of the unimodal dataset.
    """

    # Test samples
    test_cases: str = None


@dataclass
class VisualTextSample(TextSample):
    """
    The sample of the unimodal dataset.
    """

    # Images involved in the question
    # Each item is a tuple holding the image's token
    # name in the question and the path
    # For example, if the question is:
    #   <image 1> For company B, the revenue is $6,000,000.
    # Then the item to be added is ("image 1", "path/to/image")
    question_images: List[Tuple[str, str]] = None

    # Images involved in the answer
    # Each item is a list holding (image token name, images' path), which
    # is same as the one used by the `question_images`.
    # for the corresponding reasoning step,
    # i.e. one thought in the cot
    cot_images: List[List[Tuple[str, str]]] = None


@dataclass
class VisualTextCodeSample(TextCodeSample):
    """
    The sample of the unimodal dataset.
    """

    # Test visual samples
    test_visual_cases: List[Tuple[str, str]] = None


class VisualTextBase(Dataset):
    """A base class for the visual-text dataset."""

    def __init__(self, split="train", hf_dataname: str = None, config: dict = None):
        super().__init__()

        # Which part of the data to use
        self.split = split

        # The name of the dataset in the huggingface
        self.hf_dataname = hf_dataname

        # The config of the dataset
        self.config = config

        # The hf_dataset of the desired split.
        self.hf_dataset = None

        # Convert the sample to be the format required by
        # the LLMs, VLMs and so on
        self.lm_format_function = None
        # Use the visit index as the sample ID
        self.idx = 0

        # Map the dataset to the desired format
        self.map_dataset()

    def map_dataset(self):
        """Map the dataset to the desired format."""

        if self.hf_dataset is None:
            self.hf_dataset = load_dataset(self.hf_dataname, split=self.split)
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

    # A memebsership function muse be implemented
    def to_format(self, sample: dict):
        """Get the sample from the given idx."""
        raise NotImplementedError

    def batch_format(self, batch_samples: List[dict]):
        """Get the sample from the given idx."""
        # Convert dict of lists to list of dicts
        samples = [
            dict(zip(batch_samples.keys(), values))
            for values in zip(*batch_samples.values())
        ]
        samples = [self.to_format(sample) for sample in samples]

        # Convert list of dicts back to dict of lists
        return {key: [sample[key] for sample in samples] for key in samples[0].keys()}

    def save_pil_image(self, image_data, path: str, filename: str):
        """A function to save the PIL image to a file."""
        save_path = None
        if image_data is not None:
            img_format = image_data.format if image_data.format is not None else "PNG"
            extension = img_format.lower()
            if img_format.upper() == "JPEG":
                extension = "jpg"

            filename = f"{filename}.{extension}"
            save_path = f"{path}/{filename}"
            image_data.save(save_path, img_format)

            return save_path

        return save_path

    def save_example_samples(self, num_samples=15):
        """
        Save a specified number of random samples into a single JSON file.
        """
        output_dir = self.config["data_path"]
        os.makedirs(output_dir, exist_ok=True)

        # Get random samples
        sample_indices = random.sample(range(len(self.hf_dataset)), num_samples)
        samples = [self.hf_dataset[i] for i in sample_indices]

        # Prepare the filename
        filename = f"{self.split}-demo-samples.json"
        filepath = os.path.join(output_dir, filename)

        # Save all samples to a single JSON file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

        logging.info("   - Saved few samples as demos to %s", filepath)

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        """Load the sample."""
        sample = self.hf_dataset[idx]

        if len(sample["groundtruth"]) == 0:
            logging.info(
                "  !! Failed to parse gold solution: %s ", sample["cot_answer"]
            )

        # Second, make the sample to be the desired format required
        # by the LLMs and VLMs.
        if self.lm_format_function is not None:
            sample = self.lm_format_function(sample)

        return sample
