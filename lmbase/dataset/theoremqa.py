"""
Interface of the TheoremQA dataset.
"""

import os


from lmbase.dataset.base import VisualTextSample, VisualTextBase
from lmbase.identifier import MATH_SOLUTION_PROMPT


class TheoremQADataset(VisualTextBase):
    """A consistent interface for the TheoremQA dataset."""

    def __init__(
        self, split: str = "train", hf_dataname: str = None, config: dict = None
    ):

        self.data_path = config["data_path"]
        self.image_path = f"{self.data_path}/images"
        os.makedirs(self.image_path, exist_ok=True)

        super().__init__(split=split, hf_dataname=hf_dataname, config=config)

    def to_format(self, sample: dict):
        """Get the sample from the given idx."""
        self.idx += 1

        # Create the sample
        question = sample["Question"]
        question = f"{question} {MATH_SOLUTION_PROMPT}"
        image_data = sample["Picture"]
        q_image = None

        filename = f"{self.split}-Image-ID{self.idx}"
        filepath = f"{self.image_path}/{filename}.jpg"
        if os.path.exists(filepath):
            q_image = filepath
        else:
            save_path = self.save_pil_image(image_data, self.image_path, filename)
            if save_path is not None:
                q_image = save_path

        groundtruth = sample["Answer"]
        cot_answer = ""

        return VisualTextSample(
            main_id=f"ID{self.idx}",
            split=self.split,
            question=question,
            cot_answer=cot_answer,
            groundtruth=groundtruth,
            question_images=[("image", q_image)],
            sample_info={
                "dataset": self.hf_dataname,
                "answer_type": sample["Answer_type"],
            },
        )
