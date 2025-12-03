"""
Independent test for HumanEval dataset loading and formatting.

Focus:
- Code prompt and canonical solution fields
- Message formatting for code inference pipelines
"""

from lmbase.dataset import registry as dataset_registry
from lmbase import formatter


def run():
    """
    Load HumanEval `test` split and check code-focused formatting.

    Steps:
    - Retrieve dataset via registry
    - Inspect standardized sample (prompt and tests)
    - Convert to LM message format and print
    - Apply dataset-level formatting hook and fetch formatted sample
    """
    ds = dataset_registry.get(
        {
            "data_name": "humaneval",
            "data_path": "EXPERIMENT/data/humaneval",
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
