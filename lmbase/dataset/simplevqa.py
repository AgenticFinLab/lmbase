"""
Interface of the SimpleVQA dataset.

Dataset Source: https://huggingface.co/datasets/m-a-p/SimpleVQA

Description:
    A comprehensive multimodal benchmark for evaluating the factuality ability of
    multimodal large language models to answer natural language short questions.
    It covers 9 tasks (e.g., recognition of scientific, logical and medical
    contents, or tool/equipment function identification) within 9 topics, keeps
    static and timeless reference answers, and is designed to be evaluated with
    an LLM-as-a-judge scoring system.

Size: ~2,025 rows

Splits:
    - test: Test set

Features:
    - data_id: Unique identifier of the sample
    - image: Base64-encoded image string (WebP)
    - image_description: Description of the image (may be empty)
    - language: Language of the question, i.e., "CN" or "EN"
    - question: The question text
    - answer: Reference short answer
    - original_category: Original fine-grained category
    - source: Source URL of the underlying fact
    - atomic_question: Decomposed atomic question
    - atomic_fact: Decomposed atomic fact
    - vqa_category: JSON string with task/subject/entity categories

License: Apache-2.0
Language: Chinese, English
Paper: arXiv:2502.13059
"""

import io
import os
import base64
import logging

from PIL import Image

from lmbase.identifier import FINAL_SOLUTION_FLAG
from lmbase.dataset.base import VisualTextSample, VisualTextBase


class SimpleVQADataset(VisualTextBase):
    """A consistent interface for the SimpleVQA dataset."""

    def __init__(
        self, split: str = "test", hf_dataname: str = None, config: dict = None
    ):
        self.data_path = config["data_path"]
        self.image_path = os.path.join(self.data_path, "images")
        os.makedirs(self.image_path, exist_ok=True)

        super().__init__(split=split, hf_dataname=hf_dataname, config=config)

    @staticmethod
    def decode_image(image_data):
        """
        Decode the raw `image` field into a PIL image.

        The HuggingFace release stores each image as a base64-encoded WebP
        string, while other versions may expose raw bytes or decoded images.
        """
        if image_data is None:
            return None
        if isinstance(image_data, Image.Image):
            return image_data
        if isinstance(image_data, dict):
            # The HuggingFace `Image` feature, i.e., {"bytes": ..., "path": ...}
            image_data = image_data.get("bytes")
            if image_data is None:
                return None
        if isinstance(image_data, str):
            image_data = base64.b64decode(image_data)
        return Image.open(io.BytesIO(image_data))

    def to_format(self, sample: dict):
        """Get the sample from the given idx."""
        sample_id = sample["data_id"]

        question = str(sample["question"]).strip()

        question_images = []
        image_data = self.decode_image(sample.get("image"))
        if image_data is not None:
            image_token = "<image1>"
            question = f"{image_token}{question}"
            filename = f"SimpleVQA-ID{sample_id}-image1"
            save_path = self.save_pil_image(image_data, self.image_path, filename)
            if save_path is not None:
                question_images.append((image_token, save_path))
            else:
                logging.warning("Failed to save image for sample %s", sample_id)
        else:
            logging.warning("No image data for sample %s", sample_id)

        question = f"{question} {FINAL_SOLUTION_FLAG}{self.SOLUTION_FORMAT_PROMPT}"

        # SimpleVQA provides short reference answers only
        answer = str(sample["answer"]).strip()

        return VisualTextSample(
            main_id=sample_id,
            split=self.split,
            question=question,
            cot_answer=answer,
            groundtruth=answer,
            question_images=question_images,
            sample_info={
                "dataset": self.hf_dataname,
                "language": sample["language"],
                "original_category": sample["original_category"],
                "source": sample["source"],
                "atomic_question": sample["atomic_question"],
                "atomic_fact": sample["atomic_fact"],
                "vqa_category": sample["vqa_category"],
                "image_description": sample["image_description"],
            },
        )
