"""
Independent test for A-OKVQA dataset loading and formatting.

Focus:
- Visual reasoning with multiple choice options
- Message formatting including options stanza
"""

from lmbase.dataset import registry as dataset_registry
from lmbase import formatter


def run():
    """
    Load A-OKVQA `train` split and inspect multi-choice visual QA formatting.

    Steps:
    - Retrieve dataset via registry
    - Inspect standardized sample including options
    - Convert to LM message format and print
    - Apply dataset-level formatting hook and fetch formatted sample
    """
    ds = dataset_registry.get({"data_name": "aokvqa", "data_path": "EXPERIMENT/data"}, "train")
    print("Dataset:", ds)
    s = ds[0]
    print("Standardized sample:", s)
    f = formatter.map_sample(s, to_format="message")
    print("Message format:", f)
    ds.lm_format_function = lambda x: formatter.map_sample(x, to_format="message")
    print("Formatted via dataset hook:", ds[0])


if __name__ == "__main__":
    run()