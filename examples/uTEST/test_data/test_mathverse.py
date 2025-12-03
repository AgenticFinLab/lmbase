"""
Independent test for MathVerse dataset loading and formatting.

Special handling:
- Use `testmini` split in registry config
- Validates standardized sample and message formatting
"""

from lmbase.dataset import registry as dataset_registry
from lmbase import formatter


def run():
    """
    Load MathVerse with the `testmini` config and inspect formatting.

    Steps:
    - Request dataset from registry using `split="testmini"`
    - Fetch first standardized sample to verify adapter fields
    - Convert the sample to LM message format via `formatter.map_sample`
    - Attach `lm_format_function` and fetch a formatted sample through `__getitem__`
    """
    # Use special config `testmini` required by MathVerse for minimal testing
    ds = dataset_registry.get(
        {
            "data_name": "mathverse",
            "data_path": "EXPERIMENT/data/mathverse",
        },
        "testmini",
    )
    print("Dataset:", ds)

    # Inspect a standardized sample emitted by the dataset adapter
    s = ds[0]
    print("Standardized sample:", s)

    # Convert to the message format consumed by LMs
    f = formatter.map_sample(s, to_format="message")
    print("Message format:", f)

    # Enable dataset-level formatting hook for downstream consumption
    ds.lm_format_function = lambda x: formatter.map_sample(x, to_format="message")
    print("Formatted via dataset hook:", ds[0])


if __name__ == "__main__":
    run()
