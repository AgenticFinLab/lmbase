"""
Interface of the MuirBench dataset.

Dataset Source: https://huggingface.co/datasets/MUIRBENCH/MUIRBENCH

Description:
    MuirBench is a comprehensive benchmark for robust multi-image
    understanding. Each sample is a multiple-choice question over 2 to 9
    images and covers one of 12 multi-image tasks and 10 image relations,
    e.g., "Geographic Understanding", "Ordering" or "Difference Spotting".
    Unanswerable variants are included to make the evaluation more robust.
    Unlike the traditional VQA datasets, a sample cannot be answered from a
    single image, hence all the images are numbered and kept in the question.

Size: ~2,600 rows

Configurations:
    - default: The only configuration released by the dataset

Splits:
    - test: Evaluation set (~2,600 examples, this is the only split)

Features:
    - idx: Unique identifier of the sample, e.g., "0"
    - task: Multi-image task, e.g., "Geographic Understanding"
    - image_relation: Relation between the images, e.g., "Overall Similarity"
    - image_type: Type of the images, e.g., "Map"
    - question: The question text, where every image is denoted by an `<image>`
      placeholder
    - options: The list of choices, an option may also be an `<image>`
      placeholder, e.g., "<image>"
    - answer: The correct option letter, e.g., "C"
    - image_list: The images of the sample (2 to 9 images)
    - counterpart_idx: Index of the paired answerable/unanswerable variant

License: CC BY 4.0
Language: English
Paper: arXiv:2406.09411
"""

import os
import logging

from lmbase.identifier import OPTION_SOLUTION_PROMPT
from lmbase.dataset.base import VisualTextSample, VisualTextBase


# Every image of the sample is denoted by this placeholder in both the
# question and the options
IMAGE_PLACEHOLDER = "<image>"


class MUIRBenchDataset(VisualTextBase):
    """A consistent interface for the MuirBench dataset."""

    def __init__(
        self, split: str = "test", hf_dataname: str = None, config: dict = None
    ):
        self.data_path = config["data_path"]
        self.image_path = os.path.join(self.data_path, "images")
        os.makedirs(self.image_path, exist_ok=True)

        super().__init__(split=split, hf_dataname=hf_dataname, config=config)

    @staticmethod
    def number_placeholders(text: str, next_index: int):
        """
        Replace the `<image>` placeholders of `text` with numbered image
        tokens, i.e., `<image1>`, `<image2>`, ...

        The placeholders appear in the order of `image_list`, both in the
        question and in the options, hence the counter is carried through.
        """
        while IMAGE_PLACEHOLDER in text:
            next_index += 1
            text = text.replace(IMAGE_PLACEHOLDER, f"<image{next_index}>", 1)
        return text, next_index

    def to_format(self, sample: dict):
        """Get the sample from the given idx."""
        sample_id = sample["idx"]

        image_list = sample.get("image_list") or []
        question_images = []
        for i, image_data in enumerate(image_list):
            if image_data is None:
                continue
            image_token = f"<image{i + 1}>"
            filename = f"MUIRBench-{sample_id}-image{i + 1}"
            save_path = self.save_pil_image(image_data, self.image_path, filename)
            if save_path is not None:
                question_images.append((image_token, save_path))
            else:
                logging.warning(
                    "Failed to save image%d for sample %s", i + 1, sample_id
                )

        if len(question_images) == 0:
            logging.warning("No image data for sample %s", sample_id)

        # Bind the placeholders of the question and of the options to the
        # tokens of the saved images, i.e., an option may be an image itself
        question, used_images = self.number_placeholders(
            str(sample["question"]).strip(), 0
        )

        option_lines = []
        for i, option in enumerate(sample.get("options") or []):
            option_text, used_images = self.number_placeholders(
                str(option).strip(), used_images
            )
            option_lines.append(f"{chr(ord('A') + i)}. {option_text}")

        if len(option_lines) > 0:
            options_str = "\n".join(option_lines)
            question = f"{question}\nOptions:\n{options_str}"

        if used_images != len(image_list):
            logging.warning(
                "Sample %s has %d images but %d placeholders",
                sample_id,
                len(image_list),
                used_images,
            )

        question = f"{question} {OPTION_SOLUTION_PROMPT}"

        # The answer is already an option letter, e.g., "C"
        groundtruth = str(sample["answer"]).strip()

        return VisualTextSample(
            main_id=sample_id,
            split=self.split,
            question=question,
            cot_answer=groundtruth,
            groundtruth=groundtruth,
            question_images=question_images,
            sample_info={
                "dataset": self.hf_dataname,
                "task": sample.get("task"),
                "image_relation": sample.get("image_relation"),
                "image_type": sample.get("image_type"),
                "options": sample.get("options"),
                "counterpart_idx": sample.get("counterpart_idx"),
                "num_images": len(question_images),
            },
        )
