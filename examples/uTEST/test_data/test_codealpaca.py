"""
Independent test for CodeAlpaca dataset loading and formatting.

Steps:
- Load the dataset via registry
- Inspect the first standardized sample
- Convert to LM message format using `formatter.map_sample`
- Set dataset-level `lm_format_function` and fetch a formatted sample
"""

from lmbase.dataset import registry as dataset_registry
from lmbase import formatter


def run():
    """
    Load CodeAlpaca `train` split and verify text-only formatting flow.

    Steps:
    - Retrieve dataset via registry
    - Inspect standardized sample fields
    - Convert to LM message format and print
    - Attach dataset-level formatting hook and fetch formatted sample
    """
    # Load the CodeAlpaca dataset (train split) using the registry
    ds = dataset_registry.get(
        {
            "data_name": "codealpaca",
            "data_path": "EXPERIMENT/data/codealpaca",
        },
        "train",
    )
    print("Dataset:", ds)

    # Fetch the first standardized sample from the adapter
    s = ds[0]
    print("Standardized sample:", s)

    # Convert to message format expected by LMs
    f = formatter.map_sample(s, to_format="message")
    print("Message format:", f)

    # Attach formatting hook and fetch formatted sample through __getitem__
    ds.lm_format_function = lambda x: formatter.map_sample(x, to_format="message")
    print("Formatted via dataset hook:", ds[0])


if __name__ == "__main__":
    run()
