"""
Interface of the MathVerse dataset.
"""

import os
import ast
import re
import logging

from datasets import load_dataset

from lmbase.identifier import MATH_SOLUTION_PROMPT
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
            data_split = self.config["data_split"]
            self.hf_dataset = load_dataset(self.hf_dataname, data_split)
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
        sample_id = sample["sample_index"]
        # Create the sample
        question = sample["question"].strip()

        # extract all <imageN> tokens
        question_images = []
        image_tokens = re.findall(r"<image\d+>", question)
        for token in image_tokens:
            image_data = sample.get("image")
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
        options = sample.get("choices", [])
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
                "problem_index": sample.get("problem_index"),
                "problem_version": sample.get("problem_version"),
                "question_type": sample.get("question_type"),
                "metadata": sample.get("metadata"),
                "query_wo": sample.get("query_wo"),
                "query_cot": sample.get("query_cot"),
                "question_for_eval": sample.get("question_for_eval"),
            },
        )
