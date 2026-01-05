import sys
import os


from lmbase.dataset import registry as dataset_registry


def test_gpqad_loading():
    print("Testing GPQA-Diamond dataset loading...")

    # Test loading test split
    try:
        ds = dataset_registry.get(
            {
                "data_name": "gpqad",
                "data_path": "EXPERIMENT/data/gpqad",
                "solution_prompt": "\nAnswer:",
            },
            "test",
        )
        print(f"Successfully loaded GPQA-Diamond dataset with {len(ds)} samples.")

        # Check first sample
        if len(ds) > 0:
            sample = ds[0]
            print("First sample:")
            print(f"ID: {sample['main_id']}")
            print(f"Question: {sample['question']}")
            print(f"Groundtruth: {sample['groundtruth']}")

    except Exception as e:
        print(f"Failed to load GPQA-Diamond dataset: {e}")
        # Print stack trace
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_gpqad_loading()
