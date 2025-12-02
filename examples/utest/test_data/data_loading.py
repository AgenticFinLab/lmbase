"""
Comprehensive unit tests for dataset loading and formatting.

This test iterates through all datasets registered in `lmbase.dataset.registry`
and validates three core behaviors:
- Loading a dataset split using the registry
- Converting a raw sample to the message format via `formatter.map_sample`
- Applying a dataset-level `lm_format_function` to `__getitem__`

Notes:
- Some datasets may require network access to HuggingFace; failures are reported
  clearly but do not abort other tests.
- Visual datasets may save image assets to disk; this test prints basic info only.
"""

from lmbase.dataset import registry as dataset_registry
from lmbase import formatter
import traceback


def test_single_dataset(
    data_name: str,
    split: str = "train",
    data_path: str = "EXPERIMENT/data",
    **kwargs,
):
    """
    Load one dataset via the registry and validate formatting behaviors.

    Args:
        data_name (str): Registered dataset key (e.g., `gsm8k`, `math`, `humaneval`).
        split (str): Dataset split to load (e.g., `train`, `test`, `validation`).
        data_path (str): Local path for any saved assets or demo files.

    Returns:
        bool: True if loading and formatting succeed; False otherwise.
    """
    try:
        print(f"\n=== Dataset: {data_name} | Split: {split} ===")
        loaded_dataset = dataset_registry.get(
            {
                "data_name": data_name,
                "data_path": data_path,
                # for mathverse, add config_name
                "config_name": kwargs.get("config_name", split),
            },
            split,
        )
        print(f"Loaded {data_name} with {len(loaded_dataset)} samples")

        # Show the first raw sample
        sample = loaded_dataset[0]
        print("Raw sample (first):", sample)

        # 1) Convert the sample to the LM message format directly
        formatted_sample = formatter.map_sample(sample, to_format="message")
        print("Formatted sample (map_sample):", formatted_sample)

        # 2) Attach the format function to the dataset, then fetch via __getitem__
        loaded_dataset.lm_format_function = lambda x: formatter.map_sample(
            x, to_format="message"
        )
        formatted_via_dataset = loaded_dataset[0]
        print("Formatted sample (dataset lm_format_function):", formatted_via_dataset)
        return True
    except Exception as e:
        print(f"[ERROR] {data_name} ({split}) failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run data loading tests for all registered datasets on 'train' split
    all_names = list(dataset_registry.data_factory.keys())
    print("Registered datasets:", all_names)
    results = {}
    for name in all_names:
        ok = test_single_dataset(name, split="train")
        results[(name, "train")] = ok

    # Optionally, probe 'test' split where available
    for name in all_names:
        ok = test_single_dataset(name, split="test")
        results[(name, "test")] = ok

    # for mathverse, test testmini split and add config_name param
    all_names = ["mathverse"]
    for name in all_names:
        ok = test_single_dataset(name, split="testmini", config_name="testmini")
        results[(name, "testmini")] = ok

    # Summary
    print("\n=== Summary ===")
    for (name, split), ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"{name}:{split} -> {status}")
