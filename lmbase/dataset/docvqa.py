"""
Interface of the DocVQA dataset.

Dataset Source: https://huggingface.co/datasets/lmms-lab-encoder/DocVQA

Description:
    Document Visual Question Answering: each question is asked about a scanned
    document image (form, table, figure, letter, ...) and is answered with a
    short text snippet read from the document. Questions are grouped by
    documents and span several types, e.g., "figure/diagram", "form", "table".

Size: DocVQA config, ~5,349 validation rows and ~5,188 test rows

Configurations:
    - DocVQA: The DocVQA benchmark
    - InfographicVQA: A different benchmark packaged in the same repository
    Config setting in code: config name "DocVQA"

Splits:
    - validation: Labelled evaluation split (~5,349 examples)
    - test: Test split (~5,188 examples, answers not released)

Features:
    - questionId: Unique identifier of the question
    - question: The question text
    - question_types: Types of the question, e.g., ["figure/diagram"]
    - image: The document image
    - docId: Identifier of the source document
    - ucsf_document_id: Document id in the UCSF document collection
    - ucsf_document_page_no: Page number inside the document
    - answers: Reference answers from different annotators (None on test)
    - data_split: Original split name, i.e., "val" or "test"

License: Apache-2.0
Language: English
"""

import os
import logging

from datasets import load_dataset

from lmbase.dataset.base import VisualTextSample, VisualTextBase


class DocVQADataset(VisualTextBase):
    """A consistent interface for the DocVQA dataset."""

    def __init__(
        self, split: str = "validation", hf_dataname: str = None, config: dict = None
    ):
        self.data_path = config["data_path"]
        self.image_path = os.path.join(self.data_path, "images")
        os.makedirs(self.image_path, exist_ok=True)

        super().__init__(split=split, hf_dataname=hf_dataname, config=config)

    def map_dataset(self):
        """Map the dataset to the desired format."""

        if self.hf_dataset is None:
            # DocVQA is one of the configurations of this repository, the
            # generic loader without a config name does not work
            self.hf_dataset = load_dataset(
                self.hf_dataname, "DocVQA", split=self.split
            )
        super().map_dataset()

    def to_format(self, sample: dict):
        """Get the sample from the given idx."""
        sample_id = sample["questionId"]

        question = str(sample["question"]).strip()

        question_images = []
        image_token = "<image1>"
        image_data = sample.get("image")
        if image_data is not None:
            question = f"{image_token}{question}"
            filename = f"DocVQA-{sample_id}-image1"
            save_path = self.save_pil_image(image_data, self.image_path, filename)
            if save_path is not None:
                question_images.append((image_token, save_path))
            else:
                logging.warning("Failed to save image for sample %s", sample_id)
        else:
            logging.warning("No image data for sample %s", sample_id)

        # DocVQA answers are short text snippets, only the configurable
        # solution prompt is appended (empty by default)
        question = f"{question}{self.SOLUTION_FORMAT_PROMPT}"

        # Each question is annotated by several people; the first answer is used
        # as the gold answer and the full list is kept for ANLS-style evaluation.
        # The answers of the `test` split are not released, hence the empty list.
        answers = sample.get("answers") or []
        groundtruth = str(answers[0]).strip() if len(answers) > 0 else ""

        return VisualTextSample(
            main_id=sample_id,
            split=self.split,
            question=question,
            cot_answer=groundtruth,
            groundtruth=groundtruth,
            question_images=question_images,
            sample_info={
                "dataset": self.hf_dataname,
                "answers": answers,
                "question_types": sample.get("question_types"),
                "docId": sample.get("docId"),
                "ucsf_document_id": sample.get("ucsf_document_id"),
                "ucsf_document_page_no": sample.get("ucsf_document_page_no"),
                "data_split": sample.get("data_split"),
            },
        )
