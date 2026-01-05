"""
Independent test for ARC dataset loading and formatting.
"""

import sys
import os

from lmbase.dataset import registry as dataset_registry
from lmbase.identifier import MATH_SOLUTION_PROMPT


def run():
    """
    Load ARC dataset.
    """

    print("Testing ARC dataset loading...")

    # ARC splits: train, test, validation
    splits = ["train", "test", "validation"]

    # Test default subset (ARC-Challenge)
    print("\n=== Testing ARC-Challenge (Default) ===")
    for split in splits:
        print(f"\n--- Testing split: {split} ---")
        ds = dataset_registry.get(
            {
                "data_name": "arc",
                "data_path": "EXPERIMENT/data/arc",
                "SOLUTION_FORMAT_PROMPT": MATH_SOLUTION_PROMPT,
                "subset": "ARC-Challenge",
            },
            split,
        )

        print(f"Dataset loaded ({split}):", ds)
        if len(ds) > 0:
            print(ds[0])
        else:
            print(f"Dataset ({split}) is empty.")

    # Test ARC-Easy subset
    print("\n=== Testing ARC-Easy ===")
    for split in splits:
        print(f"\n--- Testing split: {split} ---")
        ds = dataset_registry.get(
            {
                "data_name": "arc",
                "data_path": "EXPERIMENT/data/arc",
                "SOLUTION_FORMAT_PROMPT": MATH_SOLUTION_PROMPT,
                "subset": "ARC-Easy",
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
