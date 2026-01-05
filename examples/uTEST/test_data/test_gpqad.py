"""
Independent test for GPQA-Diamond dataset loading and formatting.
"""

from lmbase.dataset import registry as dataset_registry
from lmbase.identifier import MATH_SOLUTION_PROMPT


def test_gpqad_loading():
    print("Testing GPQA-Diamond dataset loading...")

    # Test loading test split

    ds = dataset_registry.get(
        {
            "data_name": "gpqad",
            "data_path": "EXPERIMENT/data/gpqad",
            "SOLUTION_FORMAT_PROMPT": MATH_SOLUTION_PROMPT,
        },
        "test",
    )

    print("Dataset loaded:", ds)
    if len(ds) > 0:
        sample = ds[0]
        print(sample)


if __name__ == "__main__":
    test_gpqad_loading()
