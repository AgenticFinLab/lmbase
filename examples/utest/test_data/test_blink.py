"""
Independent test for BLINK dataset loading and formatting.

Focus:
- Multi-image visual-text sample inspection (all images saved to disk)
- Question text with one token per image
- Dataset hook application
"""

from lmbase.dataset import registry as dataset_registry
from lmbase import formatter


def run():
    """
    Load the BLINK `val` split and inspect visual-text formatting.

    Steps:
    - Retrieve dataset via registry (all 14 sub-tasks are merged)
    - Inspect standardized sample (question, image paths, answer)
    - Convert to LM message format, keeping all the images
    - Attach dataset formatting hook and fetch formatted sample
    """
    ds = dataset_registry.get(
        {
            "data_name": "blink",
            "data_path": "EXPERIMENT/data/blink",
        },
        "val",
    )
    print("Dataset:", ds)
    s = ds[0]
    print("Number of images:", len(s["question_images"]))
    print("Standardized sample:", s)
    f = formatter.map_vl_sample(s, to_format="message")
    print("Message format:", f)
    ds.lm_format_function = lambda x: formatter.map_vl_sample(x, to_format="message")
    print("Formatted via dataset hook:", ds[0])


if __name__ == "__main__":
    run()
