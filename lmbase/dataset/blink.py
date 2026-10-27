"""
Interface of the BLINK dataset.

Dataset Source: https://huggingface.co/datasets/BLINK-Benchmark/BLINK

Description:
    A multimodal benchmark that evaluates whether large vision-language models
    can perceive visual details, i.e., they "can see but not perceive". It is
    composed of 14 sub-tasks, most of which require reasoning over more than
    one image, e.g., "Counting", "Jigsaw", "Multi-view_Reasoning" or
    "Visual_Similarity". The questions are multiple choice.

Size: ~1,901 rows per split (14 sub-tasks)

Configurations:
    - This repository provides one configuration per sub-task, i.e.,
      Art_Style, Counting, Forensic_Detection, Functional_Correspondence,
      IQ_Test, Jigsaw, Multi-view_Reasoning, Object_Localization,
      Relative_Depth, Relative_Reflectance, Semantic_Correspondence,
      Spatial_Relation, Visual_Correspondence and Visual_Similarity.
    Config setting in code: all configurations are concatenated, the sub-task
    is kept in the `sub_task` field of each sample.

Splits:
    - val: Labelled evaluation split (~1,901 examples)
    - test: Test split (~1,902 examples, answers are "hidden")

Features:
    - idx: Unique identifier of the sample, e.g., "val_Counting_1"
    - question: The question text
    - sub_task: Name of the sub-task, e.g., "Counting"
    - image_1, image_2, image_3, image_4: Images of the sample, missing images
      are None (one image for Object_Localization, up to four otherwise)
    - choices: The list of choices
    - answer: The correct choice, e.g., "(B)"
    - prompt: The official prompt with the choices embedded
    - explanation: Optional explanation of the answer

License: Apache-2.0
Language: English
Paper: arXiv:2404.12390
"""

import os
import re
import logging

from datasets import load_dataset, concatenate_datasets

from lmbase.identifier import OPTION_SOLUTION_PROMPT
from lmbase.dataset.base import VisualTextSample, VisualTextBase


# One configuration per sub-task
BLINK_CONFIGS = [
    "Art_Style",
    "Counting",
    "Forensic_Detection",
    "Functional_Correspondence",
    "IQ_Test",
    "Jigsaw",
    "Multi-view_Reasoning",
    "Object_Localization",
    "Relative_Depth",
    "Relative_Reflectance",
    "Semantic_Correspondence",
    "Spatial_Relation",
    "Visual_Correspondence",
    "Visual_Similarity",
]


class BLINKDataset(VisualTextBase):
    """A consistent interface for the BLINK dataset."""

    def __init__(
        self, split: str = "val", hf_dataname: str = None, config: dict = None
    ):
        self.data_path = config["data_path"]
        self.image_path = os.path.join(self.data_path, "images")
        os.makedirs(self.image_path, exist_ok=True)

        super().__init__(split=split, hf_dataname=hf_dataname, config=config)

    def map_dataset(self):
        """Map the dataset to the desired format."""

        if self.hf_dataset is None:
            # Every sub-task is released as a separate configuration, merge
            # them so that the whole benchmark can be loaded at once
            splits = [
                load_dataset(self.hf_dataname, sub_task, split=self.split)
                for sub_task in BLINK_CONFIGS
            ]
            self.hf_dataset = concatenate_datasets(splits)
        super().map_dataset()

    @staticmethod
    def normalize_answer(answer: str):
        """Convert the answer format, e.g., "(B)", into the option letter."""
        match = re.search(r"\(([A-Za-z])\)", answer)
        if match is not None:
            return match.group(1).upper()
        return answer

    def to_format(self, sample: dict):
        """Get the sample from the given idx."""
        sample_id = sample["idx"]

        # The official prompt already contains the choices
        question = str(sample["prompt"]).strip()

        question_images = []
        image_index = 0
        for i in range(1, 5):
            image_data = sample.get(f"image_{i}")
            if image_data is None:
                continue
            image_index += 1
            image_token = f"<image{image_index}>"
            filename = f"BLINK-{sample_id}-image{image_index}"
            save_path = self.save_pil_image(image_data, self.image_path, filename)
            if save_path is not None:
                question_images.append((image_token, save_path))
            else:
                logging.warning(
                    "Failed to save image%d for sample %s", image_index, sample_id
                )

        if len(question_images) == 0:
            logging.warning("No image data for sample %s", sample_id)

        # Multi-image samples carry one token per image, in the order of
        # `image_1`, `image_2`, ...
        question = "".join([token for token, _ in question_images]) + question

        # The choices are embedded in the prompt, only the solution prompt is
        # appended so that the final option can be extracted downstream
        question = f"{question} {OPTION_SOLUTION_PROMPT}"

        raw_answer = str(sample["answer"]).strip()
        groundtruth = self.normalize_answer(raw_answer)

        return VisualTextSample(
            main_id=sample_id,
            split=self.split,
            question=question,
            cot_answer=groundtruth,
            groundtruth=groundtruth,
            question_images=question_images,
            sample_info={
                "dataset": self.hf_dataname,
                "sub_task": sample.get("sub_task"),
                "choices": sample.get("choices"),
                "answer": raw_answer,
                "explanation": sample.get("explanation"),
                "num_images": len(question_images),
            },
        )
