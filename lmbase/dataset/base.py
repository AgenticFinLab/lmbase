"""
Base dataset interfaces and utilities.

Provides unified sample containers and a visual-text dataset base class that
standardizes mapping, formatting, and lightweight asset management for use in
LLM/VLM evaluation pipelines.

Note on HuggingFace cache:
- If the dataset has already been downloaded into the local cache directory,
  HuggingFace will reuse the cached files and skip re-downloading.
- When `datasets.map` is called with `load_from_cache_file=True` and the mapping
  fingerprint matches a prior run, the map step will load cached outputs instead
  of recomputing transformations.
"""

import os
import json
import random
import logging
from typing import List, Tuple, Any
from dataclasses import dataclass

from datasets import load_dataset, config as hf_config
from torch.utils.data import Dataset
from transformers.utils import ModelOutput as FieldFrozenContainer


@dataclass
class TextSample(FieldFrozenContainer):
    """
    Canonical text-only sample schema used across datasets.

    Args:
        main_id (str): Unique identifier of the sample.
        split (str): Dataset split name (e.g., "train", "validation", "test").
        question (str): The prompt/question presented to the model.
        cot_answer (str): Chain-of-thought or explanatory answer text.
        groundtruth (str): Gold answer used for evaluation.
        sample_info (dict): Additional metadata fields specific to the dataset.
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
    Text-only sample with code-specific fields.

    Args:
        test_cases (str): Unit tests or harness code associated with the prompt.
    """

    # Test samples
    test_cases: Any = None


@dataclass
class VisualTextSample(TextSample):
    """
    Visual-text sample schema with image bindings.

    Args:
        question_images (List[Tuple[str, str]]): Pairs of (image token, image path)
            referenced in the question (e.g., ("<image1>", "/path/img1.png")).
        cot_images (List[List[Tuple[str, str]]]): Per-step image references in the
            chain-of-thought. Each inner list contains (token, path) pairs.
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
    Visual-text sample tailored for code tasks with test visuals.

    Args:
        test_visual_cases (List[Tuple[str, str]]): Visual test assets as (token, path).
    """

    # Test visual samples
    test_visual_cases: List[Tuple[str, str]] = None


class VisualTextBase(Dataset):
    """
    Base class for visual-text datasets.

    Responsibilities:
    - Load HuggingFace datasets and map raw records to standardized sample dicts
    - Provide optional formatting hook (`lm_format_function`) for downstream usage
    - Manage lightweight image saving when needed by dataset adapters
    """

    def __init__(
        self,
        split="train",
        hf_dataname: str = None,
        config: dict = None,
    ):
        super().__init__()

        # Which part of the data to use
        self.split = split

        # The name of the dataset in the huggingface
        self.hf_dataname = hf_dataname

        # The config of the dataset
        self.config = config if config is not None else {}

        # The solution prompt to append to the question
        self.solution_format_prompt = self.config.get("solution_format_prompt", "\n")

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
        """
        Map HF dataset rows to standardized sample records.

        Behavior:
        - Loads the HF split if not already available
        - Applies `batch_format` to produce the standard field schema
        - Removes original columns and writes demo samples for quick inspection
        """

        if self.hf_dataset is None:
            logging.info("   - HF cache directory: %s", hf_config.HF_DATASETS_CACHE)
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

    # A membership function must be implemented
    def to_format(self, sample: dict):
        """
        Convert a raw HF sample to the canonical schema.

        Args:
            sample (dict): Raw sample from the underlying HF dataset.

        Returns:
            dict: A standardized sample (TextSample/VisualTextSample fields).
        """
        raise NotImplementedError

    def batch_format(self, batch_samples: List[dict]):
        """
        Batch-convert raw HF rows into standardized samples.

        Args:
            batch_samples (List[dict]): A dict-of-lists representing a batch of
                HF records provided by `datasets.map` with `batched=True`.

        Returns:
            dict: Dict-of-lists containing standardized sample fields. Keys match
            the canonical schema (e.g., `main_id`, `question`, `groundtruth`).

        HuggingFace cache tip:
        - With `load_from_cache_file=True`, `datasets.map` can skip re-processing
          and load cached outputs when the mapping fingerprint (function content
          and parameters) is unchanged.
        """
        # Convert dict of lists to list of dicts
        samples = [
            dict(zip(batch_samples.keys(), values))
            for values in zip(*batch_samples.values())
        ]
        samples = [self.to_format(sample) for sample in samples]

        # Convert list of dicts back to dict of lists
        return {key: [sample[key] for sample in samples] for key in samples[0].keys()}

    def save_pil_image(self, image_data, path: str, filename: str):
        """
        Save a PIL image object to disk.

        Args:
            image_data (PIL.Image.Image): Image object to save.
            path (str): Directory where the image will be written.
            filename (str): Base filename without extension.

        Returns:
            str | None: Full saved path if successful; otherwise None.
        """
        save_path = None
        if image_data is not None:
            img_format = image_data.format if image_data.format is not None else "PNG"
            extension = img_format.lower()
            if img_format.upper() == "JPEG":
                extension = "jpg"

            filename = f"{filename}.{extension}"
            save_path = f"{path}/{filename}"

            # Skip saving if the target image already exists
            if os.path.exists(save_path):
                return save_path

            image_data.save(save_path, img_format)
            return save_path

        return save_path

    def save_example_samples(self, num_samples=15):
        """
        Save a small set of random samples into a single JSON file for inspection.

        Args:
            num_samples (int): Number of samples to write into the demo file.
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
        """Return number of samples in the mapped dataset."""
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        """
        Retrieve one mapped sample and optionally apply formatting.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            dict: Standardized sample. If `lm_format_function` is set, returns
            the formatted sample produced by that function.
        """
        sample = self.hf_dataset[idx]

        if len(sample["groundtruth"]) == 0:
            logging.info(
                "  !! Failed to parse gold solution: %s ", sample["cot_answer"]
            )

        # Apply downstream formatting when available
        if self.lm_format_function is not None:
            sample = self.lm_format_function(sample)

        return sample
