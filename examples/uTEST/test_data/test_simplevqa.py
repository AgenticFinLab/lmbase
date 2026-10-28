"""
Independent test for SimpleVQA dataset loading and formatting.

Focus:
- Visual-text sample inspection (base64 image decoded and saved to disk)
- Short-answer groundtruth in message output
- Dataset hook application
"""

from lmbase.dataset import registry as dataset_registry
from lmbase import formatter


def run():
    """
    Load SimpleVQA `test` split and inspect visual-text formatting.

    Steps:
    - Retrieve dataset via registry
    - Inspect standardized sample (question, image path, answer)
    - Convert to LM message format
    - Attach dataset formatting hook and fetch formatted sample
    """
    ds = dataset_registry.get(
        {
            "data_name": "simplevqa",
            "data_path": "EXPERIMENT/data/simplevqa",
        },
        "test",
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
