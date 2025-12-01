"""
Unified tests for block-based save and load using `BlockBasedStorer`.

This file validates two configurations to ensure correctness of:
- Block file naming: `{base}_block_{idx}.json`
- Capacity enforcement: each block stores at most `block_size` records
- Lookup behavior: `load(savename)` finds the correct record across blocks

Scenarios covered:
- `block_size = 1`: every block stores exactly one record for the same base
- `block_size = 3`: the first block stores three records, subsequent blocks store overflow
"""

import os
import json
from tempfile import TemporaryDirectory
from lmbase.utils.tools import BlockBasedStoreManager


def dump_dir(path):
    """
    Print a readable view of JSON block files in `path`.

    Notes:
    - Helps visualize how records are distributed across `{base}_block_{idx}.json` files.
    - Only prints JSON files to keep output focused on block contents.
    """
    print(f"\nDirectory: {path}")
    for name in sorted(os.listdir(path)):
        p = os.path.join(path, name)
        if os.path.isfile(p) and name.endswith(".json"):
            with open(p, "r", encoding="utf-8") as f:
                print(f"- {name}: {json.load(f)}")


def run_test_block_size_1():
    """
    Test saving and loading with `block_size = 1`.

    Expectations:
    - Each block holds exactly one record under the same base (`results`).
    - Saving three records creates three sequential blocks: `results_block_0..2.json`.
    - Loading returns the stored record when given the exact `savename` key.
    """
    with TemporaryDirectory() as tmp:
        store = BlockBasedStoreManager(tmp, file_format="json", block_size=1)

        # Save three records under base 'results'; base is parsed from `savename`
        store.save("results_1", {"acc": 0.91})
        store.save("results_2", {"acc": 0.92})
        store.save("results_3", {"acc": 0.93})

        # Expect: results_block_0.json, results_block_1.json, results_block_2.json
        files = sorted([f for f in os.listdir(tmp) if f.startswith("results_block_")])
        print("Files:", files)
        assert files == [
            "results_block_0.json",
            "results_block_1.json",
            "results_block_2.json",
        ]

        # Verify loads target the correct record by key
        assert store.load("results_2") == {"acc": 0.92}
        assert store.load("results_3") == {"acc": 0.93}

        dump_dir(tmp)


def run_test_block_size_3():
    """
    Test saving and loading with `block_size = 3`.

    Expectations:
    - First block stores three records under base `logs`.
    - The second block stores the overflow records.
    - Loading returns the stored record when given the exact `savename` key.
    """
    with TemporaryDirectory() as tmp:
        store = BlockBasedStoreManager(tmp, file_format="json", block_size=3)

        # Save five records under base 'logs'
        store.save("logs_1", {"loss": 0.31})
        store.save("logs_2", {"loss": 0.29})
        store.save("logs_3", {"loss": 0.28})
        store.save("logs_4", {"loss": 0.27})
        store.save("logs_5", {"loss": 0.26})

        # Expect: logs_block_0.json has 3 records, logs_block_1.json has 2 records
        files = sorted([f for f in os.listdir(tmp) if f.startswith("logs_block_")])
        print("Files:", files)
        assert files == ["logs_block_0.json", "logs_block_1.json"]

        # Verify loads target the correct record by key
        assert store.load("logs_3") == {"loss": 0.28}
        assert store.load("logs_4") == {"loss": 0.27}
        assert store.load("logs_5") == {"loss": 0.26}

        # Inspect contents and ensure capacities match expectations
        p0 = os.path.join(tmp, "logs_block_0.json")
        p1 = os.path.join(tmp, "logs_block_1.json")
        with open(p0, "r", encoding="utf-8") as f0, open(
            p1, "r", encoding="utf-8"
        ) as f1:
            d0 = json.load(f0)
            d1 = json.load(f1)
            assert len(d0) == 3 and set(d0.keys()) == {"logs_1", "logs_2", "logs_3"}
            assert len(d1) == 2 and set(d1.keys()) == {"logs_4", "logs_5"}

        dump_dir(tmp)


if __name__ == "__main__":
    print("Running test: block_size = 1")
    run_test_block_size_1()
    print("\nRunning test: block_size = 3")
    run_test_block_size_3()
