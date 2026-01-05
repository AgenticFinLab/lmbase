"""
Interface of the MedQA dataset.
"""

from lmbase.dataset.base import TextSample, VisualTextBase


class MedQADataset(VisualTextBase):
    """A consistent interface for the MedQA dataset."""

    def to_format(self, sample):
        """Get the sample from the given idx."""
        self.idx += 1

        # The 'data' field contains the actual content
        data = sample["data"]

        # Create the sample
        question_text = data["Question"]

        # Format options
        # Options is a dictionary like {"A": "...", "B": "..."}
        options = data["Options"]
        options_str = ""
        if options:
            # Sort keys to ensure A, B, C, D order
            sorted_keys = sorted(options.keys())
            option_lines = [f"{key}. {options[key]}" for key in sorted_keys]
            options_str = "\n".join(option_lines)

        # Combine question and options
        # We append the solution format prompt at the end
        if options_str:
            question = f"{question_text}\n{options_str}{self.SOLUTION_FORMAT_PROMPT}"
        else:
            question = f"{question_text}{self.SOLUTION_FORMAT_PROMPT}"

        groundtruth = data["Correct Option"]

        return TextSample(
            main_id=sample["id"],
            split=self.split,
            question=question,
            cot_answer="",
            groundtruth=groundtruth,
            sample_info={
                "dataset": self.hf_dataname,
                "subject_name": sample["subject_name"],
            },
        )
