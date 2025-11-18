"""
Interface of the HumanEval dataset.
"""

from fgreason.dataset.base import TextCodeSample, VisualTextBase


class HumanEvalDataset(VisualTextBase):
    """A consistent interface for the HumanEval dataset."""

    def to_format(self, sample):
        """Get the sample from the given idx."""
        self.idx += 1

        # Create the sample
        problem = sample["prompt"]
        question = f"Please complete the following function according to the given requirements and test examples.\n{problem}"
        solution = sample["canonical_solution"]
        test_str = sample["test"]
        index = test_str.find("def")
        test_cases = test_str[index:] if index != -1 else test_str

        return TextCodeSample(
            main_id=f"ID{self.idx}",
            split=self.split,
            question=question,
            cot_answer=solution,
            groundtruth=solution,
            test_cases=test_cases,
            sample_info={
                "dataset": self.hf_dataname,
                "task_id": sample["task_id"],
                "entry_point": sample["entry_point"],
            },
        )
