"""
Interface of the We-Math-2.0-Pro dataset.
"""

import os
import re
import logging

from lmbase.identifier import MATH_SOLUTION_PROMPT
from lmbase.dataset.base import VisualTextSample, VisualTextBase


class WeMath2ProDataset(VisualTextBase):
    """A consistent interface for the We-Math-2.0-Pro dataset."""

    def __init__(
        self,
        split: str = "pro",
        hf_dataname: str = None,
        config: dict = None,
    ):
        self.data_path = config["data_path"]
        self.image_path = os.path.join(self.data_path, "images")
        os.makedirs(self.image_path, exist_ok=True)

        super().__init__(split=split, hf_dataname=hf_dataname, config=config)

    def to_format(self, sample: dict):
        """Get the sample from the given idx."""
        # Create the sample
        sample_id = sample["idx"]

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

            options_found = False
            final_parsed_options = []
            final_question_text = question

            # find all "A." instances in the question
            all_a_matches = list(re.finditer(r"A\.\s*", question))

            # find the last valid options list from the end of the question
            if all_a_matches:
                for match in reversed(all_a_matches):
                    start_index = match.start()
                    # get the potential options block from the start_index to the end of the question
                    potential_options_block = question[start_index:].strip()

                    # extract individual options from the potential options block
                    individual_options = re.findall(
                        r"[A-Z]\.\s*.*?(?=\s*;\s*[A-Z]\.|$|\s+[A-Z]\.)",
                        potential_options_block,
                        re.DOTALL,
                    )

                    # extract option labels from the individual options
                    labels = []
                    for opt in individual_options:
                        m = re.match(r"([A-Z])\.", opt.strip())
                        if m:
                            labels.append(m.group(1))

                    expected_labels = [chr(ord("A") + i) for i in range(len(labels))]

                    # check if the labels are consecutive
                    if len(labels) > 1 and labels == expected_labels:
                        options_found = True

                        # parse option content
                        final_parsed_options = [
                            re.sub(r"^[A-Z]\.\s*", "", opt).strip()
                            for opt in individual_options
                        ]

                        # remove the options part from the original question
                        # because we are iterating from the end, this must be the last valid options list
                        final_question_text = question[:start_index].rstrip()
                        break

            # format the question with options if found
            if options_found:
                # create options string
                option_letters = [
                    chr(ord("A") + i) for i in range(len(final_parsed_options))
                ]
                options_str = "\n".join(
                    [
                        f"({letter}): {opt}"
                        for letter, opt in zip(option_letters, final_parsed_options)
                    ]
                )
                question = f"{final_question_text} \nOptions:\n{options_str} {MATH_SOLUTION_PROMPT}"
            else:
                # No options found in question, just add the solution prompt
                question = f"{question} {MATH_SOLUTION_PROMPT}\n"

        cot_answer = sample.get("solution", "") or ""

        groundtruth = sample["answer"].strip()

        return VisualTextSample(
            main_id=sample_id,
            split=self.split,
            question=question,
            cot_answer=cot_answer,
            groundtruth=groundtruth,
            question_images=question_images,
            sample_info={
                "dataset": self.hf_dataname,
                "question_id": sample.get("question_id"),
                "difficulty": sample.get("difficulty"),
                "knowledge points": sample.get("knowledge points"),
            },
        )
