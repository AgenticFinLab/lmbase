"""
Tools used by the whole project.
"""

import os
import re
import json
from typing import List, Any, Union, Dict

import dataclasses
from dataclasses import asdict
import torch


class BaseContainer:
    """Base container for storing information."""

    extras: Dict[str, Any] = None

    def to_dict(self):
        """
        Convert this output into a JSON-friendly dict by recursively visiting
        all fields and stringifying anything that is not directly serializable.
        Tensors are preserved in their original form.
        """

        def _ser(obj):
            if obj is None:
                return None
            if isinstance(obj, (str, int, float, bool)):
                return obj

            # if it is a tensor, directly return it
            if isinstance(obj, torch.Tensor):
                return obj

            if dataclasses.is_dataclass(obj):
                return _ser(asdict(obj))
            if isinstance(obj, dict):
                # _ser will be recursively called, so the nested tensors will also be captured by the above check
                return {str(k): _ser(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple, set)):
                # _ser will be recursively called
                return [_ser(v) for v in obj]
            if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
                try:
                    return _ser(obj.to_dict())
                except Exception:
                    return str(obj)
            if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
                try:
                    return _ser(obj.dict())
                except Exception:
                    return str(obj)
            if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
                try:
                    return _ser(obj.model_dump())
                except Exception:
                    return str(obj)
            if hasattr(obj, "to_json") and callable(getattr(obj, "to_json")):
                try:
                    j = obj.to_json()
                    if isinstance(j, str):
                        return json.loads(j)
                    return _ser(j)
                except Exception:
                    return str(obj)

            # for any other unserializable object, convert it to string
            return str(obj)

        return _ser(self)


def format_term(terminology: str):
    """
    Normalize terminology to a standard presentation.

    Args:
        terminology (str): Raw term string.

    Returns:
        str: Normalized term (title-case unless originally uppercase).
    """
    # basic conversion
    terminology = (
        terminology.replace("_", " ").replace("and", "&").replace("-", " ").rstrip()
    )
    if terminology.isupper():
        return terminology

    return terminology.title()


def remove_step_identifiers(decomposed_text):
    """
    Remove leading "Step N:" identifiers from text.

    Args:
        decomposed_text (str): Text potentially containing step labels.

    Returns:
        str: Text without "Step N:" prefixes.
    """
    return re.sub(r"Step\s*\d+\s*:\s*", "", decomposed_text)


def normalize_text(text):
    """
    Collapse sequences of whitespace into a single space and trim.

    Args:
        text (str): Input text.

    Returns:
        str: Normalized text.
    """
    text = re.sub(
        r"\s+", " ", text
    )  # Replace any whitespace sequence with a single space
    return text.strip()  # Remove leading/trailing spaces


def check_match(original_answers, decomposed_steps_list):
    """
    Compare originals to decomposed steps after removing step labels and normalizing whitespace.

    Args:
        original_answers (List[str]): Original answer strings.
        decomposed_steps_list (List[str]): Decomposed strings with step labels.

    Returns:
        List[bool]: Per-item match results.
    """
    results = []
    for original, decomposed in zip(original_answers, decomposed_steps_list):
        cleaned_decomposed = remove_step_identifiers(decomposed)
        original_normalized = normalize_text(original)
        decomposed_normalized = normalize_text(cleaned_decomposed)

        results.append(original_normalized == decomposed_normalized)
    return results


def extract_labeled_segments(
    batch_steps: List[str],
    prefixes: List[str] = None,
):
    """
    Extract content segments that follow labeled prefixes.

    Args:
        batch_steps (List[str]): Strings that may contain multiple labeled entries.
        prefixes (List[str], optional): Case-insensitive prefixes to match. Defaults to
            ["Step", "Plan"].

    Returns:
        List[List[str]]: For each input string, a list of extracted segment contents.

    Examples:
        >>> extract_labeled_segments(["Step idx: Prepare\nStep 2: Train"], ["Step"])
        [['Prepare', 'Train']]
    """
    if prefixes is None:
        prefixes = ["Step", "Plan"]

    # Build a case-insensitive alternation of prefixes, escaping any special characters
    prefix_pattern = "|".join([re.escape(p) for p in prefixes])
    pattern = (
        # Capture content following a labeled prefix and number (or 'idx') up to the next prefix
        rf"(?:{prefix_pattern})\s*(?:\d+|idx)\s*:\s*(.*?)"
        rf"(?=(?:{prefix_pattern})\s*(?:\d+|idx)\s*:\s*|$)"
    )
    # DOTALL allows newlines in captured content; IGNORECASE makes prefixes case-insensitive
    regex = re.compile(pattern, re.IGNORECASE | re.DOTALL)

    # Apply the regex to each input string and return lists of captured segments
    return [regex.findall(steps) for steps in batch_steps]


# Block-based persistence utilities
#
# This module groups records into block files named by a base key and an auto-incremented index:
#     {base}_block_{idx}.json
#
# Saving:
# - Provide a `savename` like "results_123" (base + id). The system selects the latest non-full
#   {base}_block_{idx}.json, or creates {idx+1} when full.
# Loading:
# - Given the same `savename`, the system searches {base}_block_{idx}.json files from newest to oldest.


class BlockBasedStoreManager:
    """
    Configurable block-based storage for JSON records.

    This manager groups records into multiple JSON files ("blocks") based on a common "base" name derived from the record key.
    Instead of storing all records in a single monolithic file (which becomes slow to read/write), it distributes them
    across `{base}_block_{idx}.json` files.

    Key Features:
    1. **Block-Based Storage**: Records are stored in files like `results_block_0.json`, `results_block_1.json`, etc.
       This prevents any single file from becoming too large.
    2. **Auto-Increment Index**: New blocks are created automatically when the current block reaches `block_size`.
    3. **In-Memory Info Cache**: A dictionary (`current_block_info`) tracks the content and status of the *latest active* block only.
       This optimizes memory usage by not holding the entire index in memory.
       The full index is persisted to `{base}-store-information.json`.
    4. **Sequential Access**: Designed for sequential saving without complex locking mechanisms (assumes single-writer or non-concurrent usage).
    5. **Tensor Support**: PyTorch tensors are automatically saved to separate `.pt` files, with references stored in the JSON.

    Naming and Keying:
    - Block filenames: `{base}_block_{idx}.{ext}` (e.g., `orders_block_3.json`).
    - Info filename: `{base}-store-information.json` (e.g., `orders-store-information.json`), keyed by block filename.
    - Record key: the full `savename` string is used as the key in block JSON (not parsed). Uniqueness determines whether a save is "add" or "update".
    - Base extraction: the `base` is the part before the first underscore in `savename` and decides which block family the record belongs to.

    Args:
        folder (str): Directory where block files and indices will be stored.
        file_format (str): File extension for blocks (default: "json").
        block_size (int): Maximum number of records allowed per block file (default: 1000).
    """

    def __init__(
        self,
        folder: str,
        file_format: str = "json",
        block_size: int = 1000,
    ) -> None:
        self.folder = folder
        self.file_format = file_format
        self.block_size = block_size
        # Cache only the active (latest) block info per base to save memory
        # Structure: { base: { "block_filepath": ..., "ids": [...], "size": N } }
        # Cache only the latest block info for each base in memory to save memory
        self.current_block_info: Dict[str, Dict[str, Any]] = {}
        # Info file naming pattern: {base}-store-information.json
        self.info_file_pattern = "{}-store-information.json"
        os.makedirs(self.folder, exist_ok=True)

    @staticmethod
    def _extract_base(savename: str) -> str:
        """
        Extract the base grouping key from a record name.

        The system assumes record keys follow a pattern like `base_id` (e.g., `results_123`).
        This method splits by the first underscore to isolate the `base` (e.g., `results`).
        All records sharing the same `base` will be distributed across the same set of block files.

        Args:
            savename (str): Unique record key (e.g., "experiment_v1_42").

        Returns:
            str: The extracted base name (e.g., "experiment").
        """
        # Note: Only the base (before the first underscore) is used to choose block files.
        # The remainder of `savename` is NOT parsed; the full `savename` string is used
        # as the record key inside the block JSON.
        return savename.split("_")[0]

    @staticmethod
    def _update_block_file(path: str, savename: str, data: Any) -> bool:
        """
        Read, update, and rewrite a specific block file.

        This method handles the low-level I/O:
        1. Reads existing JSON content (if any).
        2. Inserts/Updates the record under `savename`.
        3. Writes the updated dictionary back to disk atomically (via overwrite).

        Args:
            path (str): Full path to the block file.
            savename (str): The specific key for the record.
            data (Any): The JSON-serializable data to store.

        Returns:
            bool: True if a new key was added (increasing the count), False if an existing key was updated.
        """
        # The full `savename` string is the JSON key for the record within the block.
        # If the key did not previously exist, we consider this an "add" (returns True),
        # which increments the block's logical size and may trigger rotation when full.
        existing_data = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing_data = {}

        # Check if we are adding a new record or updating an old one
        was_present = savename in existing_data
        existing_data[savename] = data

        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, default=str, ensure_ascii=False, indent=2)

        return not was_present

    def _pattern(self, base: str):
        """Compile regex pattern to match block filenames for a given base (e.g., `base_block_(\\d+).json`)."""
        # Captures the numeric index `(\\d+)` of block filenames belonging to `base`.
        return re.compile(rf"^{re.escape(base)}_block_(\d+)\.{self.file_format}$")

    def _filename(self, base: str, idx: int) -> str:
        """Generate the standard filename for a block: `{base}_block_{idx}.{ext}`."""
        # Example: base="orders", idx=3, ext="json" => "orders_block_3.json"
        return f"{base}_block_{idx}.{self.file_format}"

    def _info_path(self, base: str) -> str:
        """Generate the path for the info JSON file: `{base}-store-information.json`."""
        # The info file is a dictionary keyed by block filename, with values:
        # { "ids": [...], "count": <int>, "path": "<absolute_path_to_block_file>" }
        return os.path.join(self.folder, self.info_file_pattern.format(base))

    def _get_full_info_from_disk(self, base: str) -> Dict[str, Any]:
        """
        Load the full info dictionary from disk.
        If the file is missing or corrupted, it attempts to rebuild it by scanning block files.
        """
        path = self._info_path(base)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return self._rebuild_info(base)

    def _get_latest_block_info_from_disk(
        self, base: str
    ) -> Union[Dict[str, Any], None]:
        """
        Retrieve the latest block info from the disk info file.
        This allows us to resume saving from where we left off without loading the entire history into memory.
        """
        full_info = self._get_full_info_from_disk(base)
        if not full_info:
            return None

        # The info file preserves insertion order. The last key is the latest block.
        # Since Python 3.7+, dictionaries preserve insertion order, and we write in order, so the last key is the latest block.
        latest_key = list(full_info.keys())[-1]
        data = full_info[latest_key]
        return {
            "block_filepath": data["path"],
            "ids": data["ids"],
            "size": data["count"],
        }

    def _update_info_on_disk(self, base: str, block_info: Dict[str, Any]) -> None:
        """
        Update the info file on disk with the new state of a specific block.
        This reads the full file, updates the specific block entry, and writes it back.
        """
        full_info = self._get_full_info_from_disk(base)

        # We need to map back from flattened structure to filename-keyed structure
        filepath = block_info["block_filepath"]
        filename = os.path.basename(filepath)

        # Persist block metadata back under its filename key:
        # - "ids": list of record keys present in that block
        # - "count": number of records in that block
        # - "path": absolute path to the block file
        full_info[filename] = {
            "ids": block_info["ids"],
            "count": block_info["size"],
            "path": filepath,
        }
        self._save_info_to_disk(base, full_info)

    def _rebuild_info(self, base: str) -> Dict[str, Any]:
        """
        Rebuild the info dictionary by scanning all block files.
        Useful for recovery if the info file is deleted or corrupted.
        """
        info = {}
        pattern = self._pattern(base)
        if not os.path.exists(self.folder):
            return info

        found_blocks = []
        for filename in os.listdir(self.folder):
            m = pattern.match(filename)
            if m:
                idx = int(m.group(1))
                found_blocks.append((idx, filename))

        # Sort by index to ensure insertion order in the dictionary
        found_blocks.sort(key=lambda x: x[0])

        for _, filename in found_blocks:
            path = os.path.join(self.folder, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        # Build info entry keyed by block filename with ids/count/path
                        info[filename] = {
                            "ids": list(data.keys()),
                            "count": len(data),
                            "path": path,
                        }
            except (json.JSONDecodeError, OSError, IOError):
                pass

        self._save_info_to_disk(base, info)
        return info

    def _save_info_to_disk(self, base: str, info: Dict[str, Any]) -> None:
        """Persist the info dictionary to disk."""
        path = self._info_path(base)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(info, f, default=str, ensure_ascii=False, indent=2)

    def list_blocks(self, base: str) -> List[str]:
        """
        List all existing block filenames for a given base from disk info.
        """
        info = self._get_full_info_from_disk(base)
        pattern = self._pattern(base)

        pairs = []
        for filename in info.keys():
            m = pattern.match(filename)
            if m:
                pairs.append((int(m.group(1)), filename))

        pairs.sort(key=lambda x: x[0])
        return [fname for _, fname in pairs]

    def save(self, savename: str, data: Any) -> None:
        """
        Save a record, automatically handling block allocation and info updates.

        Workflow:
        1. Prepare data (convert tensors to .pt, objects to strings).
        2. Ensure active block info is in memory (load from disk if needed).
        3. Check if current block is full; if so, create a new block.
        4. Save data to the block file.
        5. Update the info file on disk.

        Args:
            savename (str): Unique identifier for the data (e.g. "task_1").
            data (Any): The data to save.
        """
        base = self._extract_base(savename)
        # 1. Check data type and prepare value (handled in _prepare_value_for_storage)
        # Check data type. Tensors are saved as .pt with a JSON reference; other values are JSON-serialized.
        value = self._prepare_value_for_storage(savename, data)

        # 2. Check if we have active info in memory
        # Ensure we have the latest block info for this base in memory, otherwise try loading from disk.
        if base not in self.current_block_info:
            # Try to load the latest block info from disk
            # Info file uses insertion order; the last entry is the latest block.
            last_info = self._get_latest_block_info_from_disk(base)
            if last_info:
                self.current_block_info[base] = last_info
            else:
                # Initialize new block 0
                # No existing blocks found; start with `{base}_block_0.{ext}`.
                first_block_name = self._filename(base, 0)
                self.current_block_info[base] = {
                    "block_filepath": os.path.join(self.folder, first_block_name),
                    "ids": [],
                    "size": 0,
                }

        active_info = self.current_block_info[base]

        # 3. Check if we need to rotate to a new block
        # Condition: Current is full AND we are not just updating an existing ID in it
        # Rotation only happens on "adds". If you reuse the same `savename`, it's an update
        # and does not increment `size`, so rotation will not occur.
        if (
            active_info["size"] >= self.block_size
            and savename not in active_info["ids"]
        ):
            # Create next block
            current_filename = os.path.basename(active_info["block_filepath"])
            m = self._pattern(base).match(current_filename)
            current_idx = int(m.group(1)) if m else 0

            new_block_name = self._filename(base, current_idx + 1)
            active_info = {
                "block_filepath": os.path.join(self.folder, new_block_name),
                "ids": [],
                "size": 0,
            }
            self.current_block_info[base] = active_info

        # 4. Perform save
        # Persist the record into the chosen block file under key `savename`.
        path = active_info["block_filepath"]
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump({}, f, default=str, ensure_ascii=False, indent=2)

        added = self._update_block_file(path, savename, value)

        if added:
            active_info["ids"].append(savename)
            active_info["size"] += 1

        # 5. Update global info
        # Write the updated block metadata back into `{base}-store-information.json`.
        self._update_info_on_disk(base, active_info)

    def load(self, savename: str) -> Union[Any, None]:
        """
        Retrieve a record using the full info from disk.
        """
        base = self._extract_base(savename)
        # For loading, we need the full index to find where the ID is
        full_info = self._get_full_info_from_disk(base)

        # Find which block contains the savename
        # Scan the info dictionary for a block whose `ids` list contains `savename`.
        # Once found, open that block file and return the record.
        target_block = None
        for filename, block_data in full_info.items():
            if savename in block_data.get("ids", []):
                target_block = filename
                break

        if not target_block:
            return None

        path = os.path.join(self.folder, target_block)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if savename in data:
                    return self._resolve_loaded_value(data[savename])
        except (json.JSONDecodeError, OSError, IOError):
            pass

        return None

    def _prepare_value_for_storage(
        self, savename: str, data: Any, path: str = ""
    ) -> Any:
        """
        Pre-process data before JSON serialization.

        Special Handling:
        - **torch.Tensor**: Saved to `{savename}_{path}.pt`. The JSON stores a reference dict:
        `{"_type": "torch.tensor", "_path": "..."}`.
        - **Non-serializable objects**: Converted to string via `str(data)`.

        Args:
            savename (str): Record key.
            data (Any): Raw data.
            path (str): Current path in the data structure (for nested tensors).

        Returns:
            Any: JSON-safe representation.
        """
        # Handle tensors
        if isinstance(data, torch.Tensor):
            # Create a unique path for this tensor
            tensor_name = f"{savename}_{path.replace('.', '_')}" if path else savename
            tensor_dir = os.path.join(self.folder, "tensors")
            os.makedirs(tensor_dir, exist_ok=True)
            tensor_path = os.path.join(tensor_dir, f"{tensor_name}.pt")
            if not os.path.exists(tensor_path):
                torch.save(data, tensor_path)
            return {"_type": "torch.tensor", "_path": tensor_path}

        # Handle dictionaries by recursively processing each value
        elif isinstance(data, dict):
            result = {}
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                result[key] = self._prepare_value_for_storage(savename, value, new_path)
            return result

        # Handle lists/tuples by recursively processing each element
        elif isinstance(data, (list, tuple)):
            result = []
            for i, value in enumerate(data):
                new_path = f"{path}[{i}]" if path else f"[{i}]"
                result.append(
                    self._prepare_value_for_storage(savename, value, new_path)
                )
            return result

        # Handle other data types
        else:
            try:
                json.dumps(data)
                return data
            except TypeError:
                return str(data)

    def _resolve_loaded_value(self, value: Any) -> Any:
        """
        Post-process loaded JSON data to restore original objects.

        Special Handling:
        - **Tensor References**: Detects `{"_type": "torch.tensor"}` and loads the `.pt` file.

        Args:
            value (Any): Data loaded from JSON.

        Returns:
            Any: Restored object (e.g., torch.Tensor).
        """
        # Handle tensor references
        if (
            isinstance(value, dict)
            and value.get("_type") == "torch.tensor"
            and isinstance(value.get("_path"), str)
        ):
            tensor_path = value["_path"]
            if os.path.exists(tensor_path):
                return torch.load(tensor_path)

        # Handle dictionaries by recursively processing each value
        elif isinstance(value, dict):
            result = {}
            for key, val in value.items():
                result[key] = self._resolve_loaded_value(val)
            return result

        # Handle lists/tuples by recursively processing each element
        elif isinstance(value, list):
            result = []
            for val in value:
                result.append(self._resolve_loaded_value(val))
            return result

        # Return the value as is for other types
        return value
