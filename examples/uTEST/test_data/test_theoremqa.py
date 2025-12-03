"""
Independent test for TheoremQA dataset loading and formatting.

Focus:
- Visual reference to single math image
- Prompt suffix for math reasoning extraction
"""

from lmbase.dataset import registry as dataset_registry
from lmbase import formatter


def run():
    """
    Load TheoremQA `train` split and inspect visual math formatting.

    Steps:
    - Retrieve dataset via registry
    - Inspect standardized sample including a single image
    - Convert to LM message format and print
    - Apply dataset-level formatting hook and fetch formatted sample
    """
    ds = dataset_registry.get(
        {
            "data_name": "theoremqa",
            "data_path": "EXPERIMENT/data/theoremqa",
        },
        "train",
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
