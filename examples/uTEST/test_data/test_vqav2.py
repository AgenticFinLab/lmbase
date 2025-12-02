"""
Independent test for VQAv2 dataset loading and formatting.

Notes:
- Uses `validation` split
- Visual question answering format inspection
"""

from lmbase.dataset import registry as dataset_registry
from lmbase import formatter


def run():
    """
    Load VQAv2 `validation` split and inspect visual QA formatting.

    Steps:
    - Acquire dataset via registry
    - Inspect standardized sample with image bindings
    - Convert to LM message format and print
    - Enable dataset-level formatting hook and fetch formatted sample
    """
    ds = dataset_registry.get({"data_name": "vqav2", "data_path": "EXPERIMENT/data"}, "validation")
    print("Dataset:", ds)
    s = ds[0]
    print("Standardized sample:", s)
    f = formatter.map_sample(s, to_format="message")
    print("Message format:", f)
    ds.lm_format_function = lambda x: formatter.map_sample(x, to_format="message")
    print("Formatted via dataset hook:", ds[0])


if __name__ == "__main__":
    run()