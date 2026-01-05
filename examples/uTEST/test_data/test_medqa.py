"""
Independent test for MedQA dataset loading and formatting.
"""

from lmbase.dataset import registry as dataset_registry
from lmbase.identifier import MATH_SOLUTION_PROMPT


def run():
    """
    Load MedQA dataset.
    """

    print("Testing MedQA dataset loading...")

    # MedQA splits: train, test, dev
    splits = ["train", "test", "dev"]

    for split in splits:
        print(f"\n--- Testing split: {split} ---")
        ds = dataset_registry.get(
            {
                "data_name": "medqa",
                "data_path": "EXPERIMENT/data/medqa",
                "SOLUTION_FORMAT_PROMPT": MATH_SOLUTION_PROMPT,
            },
            split,
        )

        print(f"Dataset loaded ({split}):", ds)
        if len(ds) > 0:
            print(ds[0])
        else:
            print(f"Dataset ({split}) is empty.")


if __name__ == "__main__":
    run()
