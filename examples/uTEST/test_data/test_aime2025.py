"""
Independent test for AIME2025 dataset loading and formatting.
"""

import sys
import os

# Ensure we use local lmbase
sys.path.insert(0, os.getcwd())

from lmbase.dataset import registry as dataset_registry

from lmbase.identifier import MATH_SOLUTION_PROMPT



def run():
    """
    Load AIME2025 `AIME2025-I` split.
    """

    print("Testing AIME2025 dataset loading...")

    ds = dataset_registry.get(
        {
            "data_name": "aime2025",
            "data_path": "EXPERIMENT/data/aime2025",
            "SOLUTION_FORMAT_PROMPT": MATH_SOLUTION_PROMPT,
        },
        "AIME2025-I",
    )
    print("Dataset I loaded:", ds)
    if len(ds) > 0:
        s = ds[0]
        print("Sample 0:", s)
        print("Question:", s.question)
        print("Groundtruth:", s.groundtruth)

    # Test AIME2025-II loading
    print("\nTesting AIME2025-II...")
    ds2 = dataset_registry.get(
        {
            "data_name": "aime2025",
            "data_path": "EXPERIMENT/data/aime2025",
            "SOLUTION_FORMAT_PROMPT": MATH_SOLUTION_PROMPT,
        },
        "AIME2025-II",
    )
    print("Dataset II loaded:", ds2)
    if len(ds2) > 0:
        s2 = ds2[0]
        print("Sample 0:", s2)
        print("Question:", s2.question)
        print("Groundtruth:", s2.groundtruth)


if __name__ == "__main__":
    run()
