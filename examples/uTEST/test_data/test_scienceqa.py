"""
Independent test for ScienceQA dataset loading and formatting.

Notes:
- Uses `validation` split
- Visual image saving may occur; test prints standardized and message formats
"""

from lmbase.dataset import registry as dataset_registry
from lmbase import formatter


def run():
    ds = dataset_registry.get({"data_name": "scienceqa", "data_path": "EXPERIMENT/data"}, "validation")
    print("Dataset:", ds)
    s = ds[0]
    print("Standardized sample:", s)
    f = formatter.map_sample(s, to_format="message")
    print("Message format:", f)
    ds.lm_format_function = lambda x: formatter.map_sample(x, to_format="message")
    print("Formatted via dataset hook:", ds[0])


if __name__ == "__main__":
    run()