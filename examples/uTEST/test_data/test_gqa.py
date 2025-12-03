"""
Independent test for GQA dataset loading and formatting.

Notes:
- Uses `train` split
- Ensures image saving and `<imageN>` token bindings in the question
- Final answer suffix uses `FINAL_SOLUTION_FLAG`
"""

from lmbase.dataset import registry as dataset_registry
from lmbase import formatter


def run():
    """
    Load GQA `train` split and validate image binding and formatting.

    Steps:
    - Retrieve dataset via registry
    - Inspect standardized sample with `<imageN>` tokens
    - Convert to LM message format and print
    - Apply dataset-level formatting hook and fetch formatted sample
    """
    ds = dataset_registry.get(
        {
            "data_name": "gqa",
            "data_path": "EXPERIMENT/data/gqa",
        },
        "testdev",
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
