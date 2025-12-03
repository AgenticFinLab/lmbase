"""
Independent test for MathVision dataset loading and formatting.

Focus:
- Embedded `<imageN>` tokens in question and image saving
- Math solution extraction prompt suffix
"""

from lmbase.dataset import registry as dataset_registry
from lmbase import formatter


def run():
    """
    Load MathVision `train` split and validate `<imageN>` token handling.

    Steps:
    - Fetch dataset via registry
    - Inspect standardized sample with image tokens
    - Convert to LM message format and print
    - Enable dataset-level formatting hook and fetch formatted sample
    """
    ds = dataset_registry.get(
        {
            "data_name": "mathvision",
            "data_path": "EXPERIMENT/data/mathvision",
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
