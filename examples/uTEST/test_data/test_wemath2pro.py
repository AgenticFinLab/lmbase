"""
Independent test for We-Math-2.0-Pro dataset loading and formatting.

Checks:
- Registry load for `pro` split
- Standardized sample content (question, groundtruth)
- Conversion to LM message format
- Dataset hook formatting via `lm_format_function`
"""

from lmbase.dataset import registry as dataset_registry
from lmbase import formatter


def run():
    """
    Load We-Math-2.0-Pro `pro` split and verify standardized and message formats.

    Steps:
    - Fetch dataset via registry
    - Print standardized sample fields (question, groundtruth)
    - Convert sample to LM message format
    - Enable dataset-level formatting hook and fetch formatted sample
    """
    ds = dataset_registry.get(
        {
            "data_name": "wemath2pro",
            "data_path": "EXPERIMENT/data/wemath2pro",
        },
        "pro",
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
