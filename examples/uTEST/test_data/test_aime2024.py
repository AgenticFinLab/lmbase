"""
Independent test for AIME2024 dataset loading and formatting.

Notes:
- Uses `test` split
- Ensures MATH-style prompt suffix for final answer extraction
"""

from lmbase.dataset import registry as dataset_registry
from lmbase import formatter


def run():
    """
    Load AIME2024 `test` split and validate math-style formatting.

    Steps:
    - Acquire dataset via registry
    - Inspect standardized sample
    - Convert to LM message format and print
    - Apply dataset-level formatting hook and re-fetch formatted sample
    """
    ds = dataset_registry.get(
        {"data_name": "aime2024", "data_path": "EXPERIMENT/data"}, "test"
    )
    print("Dataset:", ds)
    s = ds[0]
    print("Standardized sample:", s)
    f = formatter.map_sample(s, to_format="message")
    print("Message format:", f)
    ds.lm_format_function = lambda x: formatter.map_sample(x, to_format="message")
    print("Formatted via dataset hook:", ds[0])


if __name__ == "__main__":
    run()
