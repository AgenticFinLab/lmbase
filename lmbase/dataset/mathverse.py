"""
Interface of the MathVerse dataset.
"""

import os
import re
import ast
import logging

from datasets import load_dataset

from lmbase.dataset.base import VisualTextSample, VisualTextBase


class MathVerseDataset(VisualTextBase):
    """A consistent interface for the MathVerse dataset."""

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

    def map_dataset(self):
        """Map the dataset to the desired format."""

        if self.hf_dataset is None:
            self.hf_dataset = load_dataset(self.hf_dataname, self.split)

        logging.info(
            "   - Mapping samples to lmbase format, i.e., lmbase.dataset.base.VisualTextBase"
        )

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

    def to_format(self, sample: dict):
        """Get the sample from the given idx."""
        sample_id = sample["sample_index"]
        # Create the sample
        question = sample["question"].strip()

        # extract all <imageN> tokens
        question_images = []
        image_tokens = re.findall(r"<image\d+>", question)

        # If there are no image tokens in the question but there is an image, add image tokens to the question
        image_data = sample.get("image")
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
        options = sample.get("choices", [])
        if options is None or len(options) == 0:
            question = f"{question}{self.SOLUTION_FORMAT_PROMPT}\n"
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
                question = f"{question}{self.SOLUTION_FORMAT_PROMPT}\nOptions:\n{options_str}"
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
                "problem_index": sample.get("problem_index"),
                "problem_version": sample.get("problem_version"),
                "question_type": sample.get("question_type"),
                "metadata": sample.get("metadata"),
                "query_wo": sample.get("query_wo"),
                "query_cot": sample.get("query_cot"),
                "question_for_eval": sample.get("question_for_eval"),
            },
        )
