"""
Independent test for STaRK dataset loading and formatting.

Checks:
- Registry load for `train` split
- Standardized sample content (query, groundtruth)
- Conversion to LM message format
- Dataset hook formatting via `lm_format_function`

Usage:
    # Run the test
    python examples/uTEST/test_data/test_stark.py

Expected Output:
    - Dataset: <lmbase.dataset.stark.STaRKDataset object>
    - Standardized sample: TextSample with question, groundtruth, cot_answer fields
    - Question: The query from the knowledge base
    - Groundtruth: The answer node IDs
    - Message format: List of message dicts with role and content
    - Formatted via dataset hook: Same as message format

Requirements:
    - Internet connection required for initial dataset download
    - subset config required: "amazon", "mag", or "prime"
"""

from lmbase.dataset import registry as dataset_registry
from lmbase import formatter


def run():
    """
    Load STaRK `synthesized_all_split` split and verify standardized and message formats.
    """

    ds = dataset_registry.get(
        {
            "data_name": "stark",
            "data_path": "./EXPERIMENT/data/stark",
            "subset": "prime",  # or "amazon" or "mag"
        },
        "synthesized_all_split",
    )

    print("Dataset:", ds)

    # 读取第一个样本
    s = ds[0]

    print("Standardized sample:")
    print(s)

    print("\nQuestion:")
    print(s["question"])

    print("\nGroundtruth:")
    print(s["groundtruth"])

    # 转换为 message 格式
    f = formatter.map_sample(s, to_format="message")

    print("\nMessage format:")
    print(f)

    # dataset hook
    ds.lm_format_function = lambda x: formatter.map_sample(x, to_format="message")

    print("\nFormatted via dataset hook:")
    print(ds[0])


if __name__ == "__main__":
    run()
