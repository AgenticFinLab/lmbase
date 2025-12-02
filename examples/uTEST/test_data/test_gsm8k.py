"""
Independent test for GSM8K dataset loading and formatting.

Validates:
- Registry-based loading of `train` split
- Standardized sample structure
- Conversion to LM message format
- Dataset-level formatting via `lm_format_function`
"""

from lmbase.dataset import registry as dataset_registry
from lmbase import formatter


def run():
    ds = dataset_registry.get({"data_name": "gsm8k", "data_path": "EXPERIMENT/data"}, "train")
    print("Dataset:", ds)
    s = ds[0]
    print("Standardized sample:", s)
    f = formatter.map_sample(s, to_format="message")
    print("Message format:", f)
    ds.lm_format_function = lambda x: formatter.map_sample(x, to_format="message")
    print("Formatted via dataset hook:", ds[0])


if __name__ == "__main__":
    run()