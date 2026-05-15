"""
Training and environment utility tools for TAR framework.

Functions:
    set_seed        - Set random seed for reproducibility
    get_device      - Get compute device (auto/cuda/mps/cpu)
    setup_environment - Combined seed + device setup
    count_parameters  - Count model parameters
"""

import random
import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int):
    """Set random seed for reproducibility.

    Sets seed for:
        - Python random
        - NumPy random
        - PyTorch CPU
        - PyTorch CUDA (if available)
        - cuDNN deterministic mode

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: Any = "auto") -> torch.device:
    """Get compute device based on config or auto-detection.

    Args:
        device: Device specification from config. Accepts either a string
            or an already-resolved `torch.device` (passthrough).
            - `torch.device`: returned unchanged (idempotent).
            - "auto": Auto-detect best available (cuda > mps > cpu)
            - "cuda": Force CUDA (raises error if unavailable)
            - "cuda:0", "cuda:1", etc.: Force specific GPU
            - "mps": Force MPS (Apple Silicon)
            - "cpu": Force CPU

    Returns:
        torch.device: Selected compute device.

    Raises:
        RuntimeError: If specified device is unavailable.
    """
    # Idempotent passthrough — callers may resolve once at the orchestrator
    # level and then hand the resolved object down to agents.
    if isinstance(device, torch.device):
        return device

    if device == "auto":
        if torch.cuda.is_available():
            # Select GPU with most free memory
            return torch.device(select_best_gpu())
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        # Select GPU with most free memory
        return torch.device(select_best_gpu())
    elif device.startswith("cuda:"):
        # Specific GPU requested
        gpu_id = int(device.split(":")[1])
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested but not available")
        if gpu_id >= torch.cuda.device_count():
            raise RuntimeError(
                f"GPU {gpu_id} not available (only {torch.cuda.device_count()} GPUs)"
            )
        return torch.device(f"cuda:{gpu_id}")
    elif device == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    else:
        return torch.device(device)


# Module-level cache so the "GPU Selection: ..." summary prints exactly
# once per process. Pipelines that resolve the device for both the
# environment-setup pass and per-agent inference would otherwise emit
# the same selection report multiple times.
_BEST_GPU_CACHE: Optional[str] = None


def select_best_gpu(min_memory_mb: int = 1024) -> str:
    """Select GPU with most free memory (memoized per process).

    Subsequent calls return the cached result without re-printing the
    selection summary. This guarantees that the GPU-selection log line
    appears at most once per process even when multiple components
    independently request `"auto"` device resolution.

    Args:
        min_memory_mb: Minimum free memory required in MB (default: 1024 = 1GB)

    Returns:
        str: Device string (e.g., "cuda:0", "cuda:1")

    Raises:
        RuntimeError: If no GPU has sufficient free memory.
    """
    global _BEST_GPU_CACHE
    if _BEST_GPU_CACHE is not None:
        return _BEST_GPU_CACHE

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs available")

    if num_gpus == 1:
        _BEST_GPU_CACHE = "cuda:0"
        return _BEST_GPU_CACHE

    # Get free memory for each GPU
    free_memory = []
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        # Use memory stats if available (requires torch >= 2.0)
        try:
            torch.cuda.reset_peak_memory_stats(i)
            free_mem = torch.cuda.mem_get_info(i)[0] / (1024**2)  # Convert to MB
        except:
            # Fallback: use total memory as approximation
            free_mem = props.total_memory / (1024**2)
        free_memory.append((i, free_mem))

    # Sort by free memory (descending)
    free_memory.sort(key=lambda x: x[1], reverse=True)

    # Select GPU with most free memory
    best_gpu, best_free = free_memory[0]

    if best_free < min_memory_mb:
        raise RuntimeError(
            f"No GPU has sufficient free memory. "
            f"Best GPU {best_gpu} has {best_free:.1f} MB free, "
            f"but {min_memory_mb} MB required."
        )

    print(f"GPU Selection: {num_gpus} GPUs available")
    for gpu_id, mem in free_memory:
        marker = " <-- SELECTED" if gpu_id == best_gpu else ""
        print(f"  cuda:{gpu_id}: {mem:.1f} MB free{marker}")

    _BEST_GPU_CACHE = f"cuda:{best_gpu}"
    return _BEST_GPU_CACHE


def get_dtype(dtype: Any) -> torch.dtype:
    """Resolve a dtype specification to a `torch.dtype`.

    This is the single source of truth for dtype resolution across the
    eparl/lmbase ecosystem. Callers MUST NOT re-implement string-to-dtype
    mapping locally.

    Accepted inputs:
        - `torch.dtype`           : returned unchanged.
        - "auto"                  : pick `bfloat16` if CUDA reports bf16
                                    support, else `float16` on CUDA, else
                                    `float32` on MPS/CPU.
        - "float16" / "fp16"      : `torch.float16`.
        - "bfloat16" / "bf16"     : `torch.bfloat16`.
        - "float32" / "fp32"      : `torch.float32`.

    Args:
        dtype: dtype specification (string or `torch.dtype`).

    Returns:
        torch.dtype: Resolved dtype.

    Raises:
        ValueError: If the string is not one of the supported aliases.
    """
    if isinstance(dtype, torch.dtype):
        return dtype

    if not isinstance(dtype, str):
        raise ValueError(
            f"Unsupported dtype type {type(dtype).__name__}; expected str or torch.dtype."
        )

    if dtype == "auto":
        if torch.cuda.is_available():
            # bfloat16 is preferred on Ampere+ (and many ROCm) GPUs.
            if (
                hasattr(torch.cuda, "is_bf16_supported")
                and torch.cuda.is_bf16_supported()
            ):
                return torch.bfloat16
            return torch.float16
        return torch.float32

    alias_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype not in alias_map:
        raise ValueError(
            f"Unknown dtype '{dtype}'. Supported aliases: "
            f"auto, float16/fp16, bfloat16/bf16, float32/fp32."
        )
    return alias_map[dtype]


def get_gpu_info() -> List[Tuple[int, str, float, float]]:
    """Get detailed information about all available GPUs.

    Returns:
        List of tuples: [(gpu_id, gpu_name, total_memory_gb, free_memory_gb), ...]
        Sorted by free memory (descending).
    """
    if not torch.cuda.is_available():
        return []

    num_gpus = torch.cuda.device_count()
    gpu_info = []

    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_total = torch.cuda.get_device_properties(i).total_memory / 1e9
        free, total = torch.cuda.mem_get_info(i)
        gpu_free = free / 1e9
        gpu_info.append((i, gpu_name, gpu_total, gpu_free))

    # Sort by free memory (descending)
    gpu_info.sort(key=lambda x: x[3], reverse=True)

    return gpu_info


def assign_model_devices(
    model_device_config: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    """Assign GPU devices for multiple models based on priority.

    Automatically distributes models across available GPUs based on priority.
    Higher priority (lower number) gets GPU with more free memory.

    Args:
        model_device_config: Dict mapping model name to device config.
            Each config has keys:
            - device: str, device config ("auto", "cuda:0", "cuda:1", etc.)
            - priority: int, lower number = higher priority (gets better GPU)

    Returns:
        Dict mapping model name to assigned device string (e.g., {"encoder": "cuda:0"})

    Raises:
        RuntimeError: If CUDA is not available.
        ValueError: If manually specified GPU is not available.

    Example:
        >>> config = {
        ...     "encoder": {"device": "auto", "priority": 2},
        ...     "decoder": {"device": "auto", "priority": 1},  # decoder gets best GPU
        ... }
        >>> devices = assign_model_devices(config)
        >>> print(devices)
        {"encoder": "cuda:1", "decoder": "cuda:0"}
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    num_gpus = torch.cuda.device_count()

    # Get GPU info sorted by free memory (descending)
    gpu_info = get_gpu_info()

    # Print GPU status
    print(f"[GPU] Available: {num_gpus}")
    for gpu_id, gpu_name, gpu_total, gpu_free in gpu_info:
        print(
            f"    GPU {gpu_id}: {gpu_name} (total: {gpu_total:.1f} GB, free: {gpu_free:.1f} GB)"
        )

    # Single GPU case: all models on same GPU
    if num_gpus == 1:
        result = {name: "cuda:0" for name in model_device_config}
        print(f"\n    Single GPU detected: all models on cuda:0")
        return result

    # Separate auto and manual assignments
    auto_models = []  # [(name, priority), ...]
    manual_assignments = {}  # {name: device}

    for name, cfg in model_device_config.items():
        device = cfg["device"]
        priority = cfg["priority"]
        if device == "auto":
            auto_models.append((name, priority))
        else:
            manual_assignments[name] = device

    # Validate manual GPU IDs
    for name, device in manual_assignments.items():
        gpu_id = int(device.split(":")[1])
        if gpu_id >= num_gpus:
            raise ValueError(
                f"Requested GPU {gpu_id} for '{name}' not available. "
                f"Only {num_gpus} GPUs found."
            )

    # Track used GPUs (by manual assignment)
    used_gpu_ids = set(int(d.split(":")[1]) for d in manual_assignments.values())

    # Sort auto models by priority (lower number = higher priority)
    auto_models.sort(key=lambda x: x[1])

    # Assign auto models to available GPUs
    result = dict(manual_assignments)
    available_gpus = [g for g in gpu_info if g[0] not in used_gpu_ids]

    print(f"\n    Auto GPU assignment:")
    for name, priority in auto_models:
        if available_gpus:
            # Get best available GPU
            best_gpu = available_gpus.pop(0)  # Already sorted by free memory
            device = f"cuda:{best_gpu[0]}"
            result[name] = device
            print(
                f"      {name} (priority={priority}) -> {device} ({best_gpu[3]:.1f} GB free)"
            )
        else:
            # No available GPUs, share with existing assignment
            # Find the least loaded GPU
            best_gpu = gpu_info[0]
            device = f"cuda:{best_gpu[0]}"
            result[name] = device
            print(
                f"      {name} (priority={priority}) -> {device} (shared, {best_gpu[3]:.1f} GB free)"
            )

    # Print manual assignments
    if manual_assignments:
        print(f"\n    Manual GPU assignment:")
        for name, device in manual_assignments.items():
            print(f"      {name} -> {device}")

    return result


def setup_environment(env_cfg: dict) -> torch.device:
    """Setup training environment from config.

    Handles:
        1. Random seed setting (Python, NumPy, PyTorch, CUDA)
        2. Device selection (auto/cuda/mps/cpu)

    Args:
        env_cfg: Environment config dict with keys:
            - seed (int): Random seed for reproducibility
            - device (str): Device specification ("auto", "cuda", "mps", "cpu")

    Returns:
        torch.device: Selected compute device.

    Example:
        env_cfg = {"seed": 42, "device": "auto"}
        device = setup_environment(env_cfg)
    """
    # Set random seed
    seed = env_cfg["seed"]
    set_seed(seed)

    # Get device
    device_str = env_cfg["device"]
    device = get_device(device_str)

    return device


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in a model.

    Args:
        model: PyTorch model.
        trainable_only: If True, count only trainable parameters.

    Returns:
        int: Number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def collate_fn_text(batch) -> List[str]:
    """Extract text from dataset samples.

    Supports common dataset formats with 'question' or 'problem' fields.

    Args:
        batch: List of dataset samples (dicts or other).

    Returns:
        List[str]: Extracted text strings.
    """
    texts = []
    for sample in batch:
        if isinstance(sample, dict):
            if "question" in sample:
                texts.append(sample["question"])
            elif "problem" in sample:
                texts.append(sample["problem"])
            elif "text" in sample:
                texts.append(sample["text"])
            else:
                texts.append(str(sample))
        else:
            texts.append(str(sample))
    return texts


def decode_logits_to_text(
    logits: torch.Tensor,
    tokenizer,
    original_texts: List[str] = None,
    attention_mask: torch.Tensor = None,
) -> dict:
    """Decode logits to text and compare with original.

    Restoration procedure:
        logits [B, L, V=50257] -> argmax(dim=-1) -> pred_ids [B, L]
        pred_ids [B, L] -> tokenizer.decode() -> List[str] texts

        V=50257 is GPT2's vocabulary size:
        - Each position has 50257 logits (one per token)
        - argmax over dim=-1 selects the most likely token ID
        - tokenizer.decode converts token IDs back to text

    Args:
        logits: Decoder output [B, L, V]
        tokenizer: Tokenizer for decoding (e.g., GPT2Tokenizer)
        original_texts: Optional original texts for comparison
        attention_mask: Optional mask [B, L], 1=valid, 0=pad
            If provided, only decodes up to actual sequence length

    Returns:
        dict with keys:
            - pred_ids: List[List[int]] predicted token IDs (JSON-serializable)
            - pred_texts: List[str] decoded texts
            - original_texts: List[str] original input texts (if provided)
            - comparisons: List[dict] with original/reconstructed pairs
    """
    # logits [B, L, V] -> argmax(dim=-1) -> pred_ids [B, L]
    pred_ids_tensor = logits.argmax(dim=-1)

    # Convert to list for JSON serialization
    pred_ids = pred_ids_tensor.cpu().tolist()

    # pred_ids [B, L] -> tokenizer.decode() -> List[str]
    pred_texts = []
    for i in range(pred_ids_tensor.shape[0]):
        if attention_mask is not None:
            # Find actual sequence length (up to first pad position)
            mask = attention_mask[i].cpu()
            seq_len = mask.sum().item()
            # Only decode up to actual length
            ids = pred_ids_tensor[i, :seq_len]
        else:
            ids = pred_ids_tensor[i]
        text = tokenizer.decode(ids, skip_special_tokens=True)
        pred_texts.append(text)

    result = {
        "pred_ids": pred_ids,
        "pred_texts": pred_texts,
    }

    # Add original texts and comparisons if provided
    if original_texts is not None:
        result["original_texts"] = original_texts
        comparisons = []
        for i, (orig, pred) in enumerate(zip(original_texts, pred_texts)):
            comparisons.append(
                {
                    "index": i,
                    "original": orig,
                    "reconstructed": pred,
                }
            )
        result["comparisons"] = comparisons

    return result


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Find the latest checkpoint file in directory.

    Args:
        checkpoint_dir: Directory containing checkpoint-*.pt files

    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    checkpoint_files = list(checkpoint_dir.glob("checkpoint-*.pt"))
    if not checkpoint_files:
        return None

    # Sort by global_step (extracted from filename)
    def get_step_from_ckpt(f):
        # Format: checkpoint-epoch{e}-step{s}-global{g}.pt
        name = f.stem
        parts = name.split("-")
        for p in parts:
            if p.startswith("global"):
                return int(p[6:])
        return 0

    checkpoint_files.sort(key=get_step_from_ckpt, reverse=True)
    return checkpoint_files[0]


def resume_from_checkpoint(
    checkpoint_path: Path,
    models: Dict[str, nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: str,
    log_dir: Optional[Path] = None,
) -> Tuple[int, int, Dict[str, Any]]:
    """Resume training from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        models: Dict mapping model names to model instances
            - For ed_train: {"encoder": encoder, "decoder": decoder}
            - For eqd_train: {"encoder": encoder, "quantizer": quantizer, "decoder": decoder}
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        device: Device to load models to
        log_dir: Optional log directory to load training history

    Returns:
        Tuple of (start_epoch, global_step, history)
        - start_epoch: Epoch to resume from (0-indexed)
        - global_step: Global step to resume from
        - history: Training history dict
    """
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Load model weights
    for name, model in models.items():
        state_key = f"{name}_state_dict"
        if state_key in ckpt:
            model.load_state_dict(ckpt[state_key])

    # Load optimizer and scheduler
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    # Extract training state
    start_epoch = ckpt["epoch"] - 1  # Will increment in loop
    global_step = ckpt["global_step"]

    # Load training history if exists
    history = {"config": {}, "steps": []}
    if log_dir is not None:
        history_path = log_dir / "training_history.json"
        if history_path.exists():
            with open(history_path, "r") as f:
                history = json.load(f)

    return start_epoch, global_step, history
