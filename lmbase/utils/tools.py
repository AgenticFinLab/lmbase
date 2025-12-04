"""
Tools used by the whole project.
"""

import os
import re
import json
import fcntl
import time
import random
from typing import List, Any, Union
import torch


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
#   `{base}_block_{idx}.json`, or creates `{idx+1}` when full.
# Loading:
# - Given the same `savename`, the system searches `{base}_block_{idx}.json` files from newest to oldest.


class BlockBasedStoreManager:
    """
    Configurable block-based storage for JSON records.

    Groups records under base-named block files and manages auto-incremented indices.

    Args:
        folder (str): Directory to store block files.
        file_format (str): File extension (defaults to 'json').
        block_size (int): Maximum records per block (defaults to 1000).

    Methods:
        save(savename: str, data: Any) -> None: Save/update a record into a block.
        load(savename: str) -> Any | None: Load a record across blocks.
        list_blocks(base: str) -> List[str]: List block filenames for a base, ascending by index.
        next_block_name(base: str) -> str: Compute next block filename for a base.
        find_nonfull_block(base: str) -> str: Find the newest non-full block for a base.
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
        os.makedirs(self.folder, exist_ok=True)

    @staticmethod
    def _extract_base(savename: str) -> str:
        """
        Extract base name from savename by removing the trailing underscore and id.

        Args:
            savename (str): String in the format 'base_id', e.g., 'results_123'.

        Returns:
            str: The base part of the savename, e.g., 'results'.
        """
        return savename.split("_")[0]

    @staticmethod
    def _safe_file_operation(
        file_path: str, operation_func, *args, max_retries: int = 10, **kwargs
    ):
        """
        Run a file operation with retries for concurrent environments.

        Args:
            file_path (str): Target file path.
            operation_func (Callable[..., Any]): Function that performs the operation.
            *args: Positional arguments passed to `operation_func`.
            max_retries (int): Maximum retry attempts. Defaults to 10.
            **kwargs: Keyword arguments passed to `operation_func`.

        Returns:
            Any: The return value from `operation_func`.
        """
        for attempt in range(max_retries):
            try:
                return operation_func(file_path, *args, **kwargs)
            except (OSError, IOError, json.JSONDecodeError) as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(random.uniform(0.1, 0.5))
                continue

    @staticmethod
    def _update_block_file(path: str, savename: str, data: Any) -> None:
        """
        Update a block file with a new record under key `savename`.

        Args:
            path (str): Block file path to update.
            savename (str): Record key.
            data (Any): JSON-serializable content to write.
        """
        with open(path, "r+", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0)
                existing_data = json.load(f)
                existing_data[savename] = data
                f.seek(0)
                f.truncate()
                json.dump(existing_data, f, default=str, ensure_ascii=False, indent=2)
            finally:
                pass

    def _pattern(self, base: str):
        return re.compile(rf"^{re.escape(base)}_block_(\d+)\.{self.file_format}$")

    def _filename(self, base: str, idx: int) -> str:
        return f"{base}_block_{idx}.{self.file_format}"

    def list_blocks(self, base: str) -> List[str]:
        """
        List block files for a base, ascending by block index.

        Args:
            base (str): Base name.

        Returns:
            List[str]: Filenames sorted ascending by index.
        """
        # Accumulate (index, filename) pairs for this base
        pairs: List[tuple[int, str]] = []
        pattern = self._pattern(base)
        for filename in os.listdir(self.folder):
            m = pattern.match(filename)
            if m:
                pairs.append((int(m.group(1)), filename))
        # Sort ascending by index to provide stable ordering
        pairs.sort(key=lambda x: x[0])
        return [fname for _, fname in pairs]

    def next_block_name(self, base: str) -> str:
        """
        Compute next block filename for a base.

        Args:
            base (str): Base name.

        Returns:
            str: Next filename `{base}_block_{idx}.{ext}`.
        """
        blocks = self.list_blocks(base)
        if not blocks:
            return self._filename(base, 0)
        pattern = self._pattern(base)
        # Take the last block and extract its numeric index
        last = blocks[-1]
        idx = int(pattern.match(last).group(1))
        return self._filename(base, idx + 1)

    def find_nonfull_block(self, base: str) -> str:
        """
        Find newest non-full block for a base.

        Args:
            base (str): Base name.

        Returns:
            str: Filename of a non-full block; empty string if none.
        """
        blocks = self.list_blocks(base)
        for filename in reversed(blocks):
            path = os.path.join(self.folder, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    # Parse and check occupancy; treat malformed JSON as non-candidate
                    data = json.load(f)
                    if isinstance(data, dict) and len(data) < self.block_size:
                        return filename
            except (json.JSONDecodeError, FileNotFoundError, OSError, IOError):
                # Ignore broken/missing files and continue searching
                continue
        return ""

    def save(self, savename: str, data: Any) -> None:
        """
        Save a record to base-named blocks.

        Args:
            savename (str): Record key (e.g., "results_123").
            data (Any): JSON-serializable content.
        """
        base = self._extract_base(savename)
        value = self._prepare_value_for_storage(savename, data)
        # Prefer the newest non-full block for this base; otherwise create a new one
        target = self.find_nonfull_block(base)
        if not target:
            target = self.next_block_name(base)
            path = os.path.join(self.folder, target)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {savename: value}, f, default=str, ensure_ascii=False, indent=2
                )
        else:
            path = os.path.join(self.folder, target)
            # Execute the update under retry protection to mitigate transient IO errors
            self._safe_file_operation(
                path, self._update_block_file, savename, value, max_retries=10
            )

    def load(self, savename: str) -> Union[Any, None]:
        """
        Load a record across base-named blocks.

        Args:
            savename (str): Record key (e.g., "results_123").

        Returns:
            Any | None: Record content when found; otherwise None.
        """
        base = self._extract_base(savename)
        pattern = self._pattern(base)
        # Collect all blocks for this base; if none, indicate absence
        files = [name for name in os.listdir(self.folder) if pattern.match(name)]
        if not files:
            return None
        use_new = all(pattern.match(n) for n in files)
        # Sort by descending index (newest first) for efficient lookup
        ordered = (
            sorted(files, key=lambda n: int(pattern.match(n).group(1)), reverse=True)
            if use_new
            else sorted(files, reverse=True)
        )
        for name in ordered:
            path = os.path.join(self.folder, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    # Return the first match encountered (newest block wins)
                    data = json.load(f)
                    if savename in data:
                        return self._resolve_loaded_value(data[savename])
            except (json.JSONDecodeError, OSError, IOError):
                # Skip unreadable files and continue
                continue
        return None

    def _prepare_value_for_storage(self, savename: str, data: Any) -> Any:
        """
        Convert input data into a JSON-serializable value. If the data is a torch.Tensor
        and torch is available, save it to a separate .pt file and store a reference.

        Args:
            savename (str): Record key.
            data (Any): Input data to persist.

        Returns:
            Any: JSON-serializable value (possibly a reference dict).
        """
        if isinstance(data, torch.Tensor):
            tensor_path = os.path.join(self.folder, f"{savename}.pt")
            # If the target file already exists, do not re-save
            if not os.path.exists(tensor_path):
                torch.save(data, tensor_path)
            return {"_type": "torch.tensor", "_path": tensor_path}
        try:
            json.dumps(data)
            return data
        except TypeError:
            return str(data)

    def _resolve_loaded_value(self, value: Any) -> Any:
        """
        Resolve a stored value into its runtime object. If the value is a tensor reference,
        load it via torch.load when available.

        Args:
            value (Any): Stored value from JSON.

        Returns:
            Any: Runtime data (e.g., torch.Tensor) or the original value.
        """
        if (
            isinstance(value, dict)
            and value.get("_type") == "torch.tensor"
            and isinstance(value.get("_path"), str)
        ):
            tensor_path = value["_path"]
            if os.path.exists(tensor_path):
                return torch.load(tensor_path)
        return value
