import os
import shutil
import json
import glob
from lmbase.utils.tools import BlockBasedStoreManager


def run_test():
    folder = "EXPERIMENT/uTEST/BlockManager"
    if os.path.exists(folder):
        shutil.rmtree(folder)

    # Initialize with small block size to force multiple blocks
    # Initialize Manager with a small block_size=3 to easily trigger new block creation testing
    manager = BlockBasedStoreManager(folder=folder, block_size=3)

    """
    Test saving 29 items with block_size=3.
    Expected:
    - 29 items total.
    - Blocks 0-8 (9 blocks) should be full (3 items each). Total 27 items.
    - Block 9 should have 2 items (items 28, 29).
    - Total blocks: 10 (0 to 9).
    """
    base_name = "testdata"
    total_items = 29

    print(f"\n[Test] Starting save of {total_items} items with block_size=3...")

    for i in range(total_items):
        key = f"{base_name}_{i}"
        # Save normal dictionary data
        data = {"index": i, "content": f"value_{i}"}

        manager.save(key, data)

        # Verify current block info in memory matches expectation
        # Verify if current_block_info in memory matches expectations
        active_info = manager.current_block_info.get(base_name)
        assert active_info is not None

        # Calculate expected block index and size
        # Calculate expected block index and current size
        expected_block_idx = i // 3
        expected_size = (i % 3) + 1

        current_block_path = active_info["block_filepath"]
        assert current_block_path.endswith(
            f"{base_name}_block_{expected_block_idx}.json"
        )
        assert active_info["size"] == expected_size
        assert key in active_info["ids"]

    print("[Test] Finished saving. Verifying final state...")

    # 1. Verify Info File
    # 1. Verify if the info file on disk contains correct information for all blocks
    info_path = os.path.join(folder, f"{base_name}-store-information.json")
    assert os.path.exists(info_path), "Info file should exist"

    with open(info_path, "r") as f:
        full_info = json.load(f)

    # Should have 10 blocks: 0 to 9
    assert len(full_info) == 10

    # Verify block sequence and content counts
    sorted_blocks = sorted(
        full_info.keys(), key=lambda x: int(x.split("_")[-1].split(".")[0])
    )

    for idx, block_filename in enumerate(sorted_blocks):
        assert block_filename == f"{base_name}_block_{idx}.json"
        block_data = full_info[block_filename]
        if idx < 9:
            assert block_data["count"] == 3, f"Block {idx} should be full"
        else:
            assert block_data["count"] == 2, f"Block {idx} (last) should have 2 items"

    # 2. Verify Data Loading (Random Access)
    # 2. Verify data loading (random access testing)
    print("[Test] Verifying data loading...")

    # Load item 0 (Block 0)
    val_0 = manager.load(f"{base_name}_0")
    assert isinstance(val_0, dict)
    assert val_0["index"] == 0

    # Load item 10 (Block 3, Index 1)
    val_10 = manager.load(f"{base_name}_10")
    assert isinstance(val_10, dict)
    assert val_10["index"] == 10

    # Load item 28 (Block 9, Index 1, Dict)
    val_28 = manager.load(f"{base_name}_28")
    assert isinstance(val_28, dict)
    assert val_28["index"] == 28

    # 3. Verify Filling the Last Block
    # 3. Verify filling the last block
    print("[Test] Filling the last block (Item 29)...")
    manager.save(f"{base_name}_29", {"data": "fill"})

    active_info = manager.current_block_info[base_name]
    assert active_info["block_filepath"].endswith(f"{base_name}_block_9.json")
    assert active_info["size"] == 3

    # Check info file again
    with open(info_path, "r") as f:
        full_info = json.load(f)
    assert full_info[f"{base_name}_block_9.json"]["count"] == 3

    # 4. Verify Creating New Block (Item 30)
    # 4. Verify automatic creation of new block when full (Item 30)
    print("[Test] Creating new block (Item 30)...")
    manager.save(f"{base_name}_30", {"data": "new_block"})

    active_info = manager.current_block_info[base_name]
    assert active_info["block_filepath"].endswith(f"{base_name}_block_10.json")
    assert active_info["size"] == 1

    # Verify persistence
    with open(info_path, "r") as f:
        full_info = json.load(f)
    assert f"{base_name}_block_10.json" in full_info

    print("[Test] All checks passed.")


if __name__ == "__main__":
    run_test()
