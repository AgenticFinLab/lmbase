"""
Independent test for AIME2025 dataset loading and formatting.
"""

import sys
import os

# Ensure we use local lmbase
sys.path.insert(0, os.getcwd())

from lmbase.dataset import registry as dataset_registry


def run():
    """
    Load AIME2025 `AIME2025-I` split.
    """

    print("Testing AIME2025 dataset loading...")

    ds = dataset_registry.get(
        {
            "data_name": "aime2025",
            "data_path": "EXPERIMENT/data/aime2025",
        },
        "AIME2025-I",
    )
    print("Dataset I loaded:", ds)
    print("Sample 0:", ds[0])

    # Test AIME2025-II loading
    print("\nTesting AIME2025-II...")
    ds2 = dataset_registry.get(
        {
            "data_name": "aime2025",
            "data_path": "EXPERIMENT/data/aime2025",
        },
        "AIME2025-II",
    )
    print("Dataset II loaded:", ds2)
    print("Sample 0:", ds2[0])


if __name__ == "__main__":
    run()
