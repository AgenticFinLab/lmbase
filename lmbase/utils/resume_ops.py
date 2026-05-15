"""Resume I/O helpers for nlcpV4 training scripts.

This module is the single place that knows how to preserve the
observability artefacts of a previous training run when a new run
resumes from a checkpoint. Three concerns are covered:

1. SwanLab run continuity via ``logs/<exp>/swanlab.json``.
   A fresh run writes the file once with the newly-allocated run id;
   every subsequent resume records a ``resume_events`` entry and bumps
   ``resume_count``. The file is the on-disk source of truth consulted
   when the operator does NOT pass ``--swanlab-id`` explicitly.

2. Rotation of log files that CANNOT be safely appended.
   ``training.log`` (python logging FileHandler) and
   ``terminal_output.jsonl`` (line-oriented JSON) are renamed to
   ``<stem>.part<N>.<ext>`` on every resume, with N monotonically
   increasing across resumes. The CURRENT run always writes to the
   un-suffixed name, so ``tail -f training.log`` during an active run
   works without knowing about parts.

3. Reload of history JSON files that CAN be extended.
   ``training_history.json``, ``eval_history.json``, and
   ``eval_sample_history.json`` are plain lists of rows; on resume we
   load them back into memory so the new run appends to the existing
   history rather than clobbering it when the first epoch-end rewrite
   fires.

The module is intentionally free of any swanlab or top-level torch
imports so it can be unit-tested and imported by the launcher without
pulling heavy dependencies. The schema-validation peek inside
``find_latest_checkpoint`` lazy-imports torch ONLY when an actual
resume is performed (never during launcher import).
"""

from __future__ import annotations

import datetime
import json
import re
import sys
from pathlib import Path
from typing import Optional

SWANLAB_META_FILENAME = "swanlab.json"

# All periodic / best / epoch-end checkpoints end with
# ``-epoch<N>-step<M>.pt`` — see ``save_checkpoint`` callers in
# ``train_builder.py``. Auto-discovery keys off this suffix so a new
# checkpoint naming variant only needs to keep the trailing tag.
_CKPT_PATTERN = re.compile(r"^checkpoint.*-epoch(\d+)-step(\d+)\.pt$")

# Keys every nlcpV4 trainer-written checkpoint MUST carry for
# ``load_checkpoint`` to be able to fully restore (model + optimizer +
# scheduler + position). ``loss`` is intentionally NOT in this list:
# legacy ``checkpoint_best_eval-*.pt`` files (commit 1a510ef) wrote
# only ``eval_loss``, and ``load_checkpoint`` already accepts either
# key with an ``inf`` fallback. The schema-validation peek only
# rejects checkpoints that cannot be RESTORED at all.
_REQUIRED_CKPT_KEYS: tuple[str, ...] = (
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "epoch",
    "step",
)


def _is_schema_valid_checkpoint(
    path: Path,
    required_keys: tuple[str, ...] = _REQUIRED_CKPT_KEYS,
) -> bool:
    """Peek a torch checkpoint and return True iff all required keys exist.

    Lazy-imports torch so the launcher (which only needs
    ``CheckpointNotFoundError``) does not pay the torch import cost.
    A fully unloadable / malformed pickle is treated as invalid (False)
    and a one-line warning is emitted on stderr; the caller is expected
    to skip and try the next-best candidate.

    Note: torch.load has no incremental peek API, so this fully
    materialises the dict (mapped to CPU). For the eventually-chosen
    file this means a ~2x load cost on resume, which is acceptable for
    a one-time event; rejected files cost only their single peek.
    """
    import torch  # local: keep module import cheap for the launcher

    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001 — any load failure = invalid
        print(
            f"[resume] WARN: failed to peek {path.name}: {exc!r} "
            f"— treating as schema-incompatible.",
            file=sys.stderr,
        )
        return False
    if not isinstance(ckpt, dict):
        print(
            f"[resume] WARN: {path.name} is not a dict checkpoint "
            f"(got {type(ckpt).__name__}) — skipping.",
            file=sys.stderr,
        )
        return False
    missing = [k for k in required_keys if k not in ckpt]
    if missing:
        print(
            f"[resume] WARN: {path.name} missing required keys "
            f"{missing} — skipping (likely legacy / non-trainer file).",
            file=sys.stderr,
        )
        return False
    return True


class CheckpointNotFoundError(FileNotFoundError):
    """Raised when ``--resume`` is set but no loadable checkpoint exists.

    A distinct subclass makes it trivial to distinguish "operator
    asked to resume but the directory is empty / missing" from a
    generic I/O error inside torch.load.
    """


def find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    """Return the most-recent checkpoint file inside ``checkpoint_dir``.

    "Most recent" = the file whose ``epoch<N>-step<M>`` tag has the
    highest ``(step, epoch)`` tuple. ``global_step`` is the monotonic
    authority during training; ``epoch`` only breaks ties when two
    files share the same step (rare, e.g. an epoch-end checkpoint
    taken at the same step as a periodic one).

    Matches any filename shaped like
    ``checkpoint*-epoch<N>-step<M>.pt`` so best / best_eval / periodic
    / epoch-end variants are ALL considered. This is deliberate: on
    resume we want to continue from the latest STATE, whatever tag it
    happens to carry — a crash right after a best_eval save is a
    perfectly valid anchor for the next run.

    Schema-validated walk: candidates are visited newest → oldest and
    each one is peeked via ``_is_schema_valid_checkpoint``. Files that
    cannot be loaded as a dict, or that are missing required keys
    (``model_state_dict`` / ``optimizer_state_dict`` /
    ``scheduler_state_dict`` / ``epoch`` / ``step``), are skipped
    with a stderr WARN and the next-best candidate is tried. This
    keeps resume robust to historical schema drift — e.g. legacy
    ``checkpoint_best_eval-*.pt`` written by ``commit 1a510ef`` had a
    different (eval-only) shape and would otherwise crash the loader.

    Raises ``CheckpointNotFoundError`` if the directory is missing,
    contains zero matching files, OR every matching file fails the
    schema check. Fail-loud is strictly better than a silent
    fresh-start that clobbers the operator's expectation of
    continuity.
    """
    if not checkpoint_dir.exists():
        raise CheckpointNotFoundError(
            f"--resume requested but checkpoint_dir does not exist: "
            f"{checkpoint_dir}. Nothing to resume from."
        )
    candidates: list[tuple[int, int, Path]] = []
    for p in checkpoint_dir.iterdir():
        if not p.is_file():
            continue
        m = _CKPT_PATTERN.match(p.name)
        if m:
            candidates.append((int(m.group(2)), int(m.group(1)), p))
    if not candidates:
        raise CheckpointNotFoundError(
            f"--resume requested but no 'checkpoint*-epoch<N>-step<M>.pt' "
            f"files were found in {checkpoint_dir}. Either the training "
            f"run has not produced any checkpoints yet, or the directory "
            f"is misconfigured in YAML (log.checkpoint_path)."
        )
    # candidates elements are (step, epoch, path); sort ascending by
    # (step, epoch). We do not break ties on the filename itself
    # because lexicographic order of tags has no meaningful ranking
    # (best vs clean vs epoch-end).
    candidates.sort()
    # Walk newest → oldest; first SCHEMA-VALID candidate wins. This
    # shields resume from heterogeneous schema drift: e.g. a legacy
    # ``checkpoint_best_eval-*.pt`` written by an older code revision
    # (lacking model/optimizer/scheduler state in the unified shape)
    # is silently passed over in favour of the next-best file.
    skipped: list[Path] = []
    for _step, _epoch, path in reversed(candidates):
        if _is_schema_valid_checkpoint(path):
            if skipped:
                print(
                    f"[resume] picked {path.name} after skipping "
                    f"{len(skipped)} schema-incompatible candidate(s): "
                    f"{[p.name for p in skipped]}",
                    file=sys.stderr,
                )
            return path
        skipped.append(path)
    raise CheckpointNotFoundError(
        f"--resume requested but no SCHEMA-VALID checkpoint was found in "
        f"{checkpoint_dir}. {len(skipped)} candidate(s) matched the name "
        f"pattern but all lacked required keys {_REQUIRED_CKPT_KEYS}: "
        f"{[p.name for p in skipped]}."
    )


# ── Part-suffix rotation ──────────────────────────────────────────────


def rotate_if_exists(path: Path) -> Optional[Path]:
    """Rename ``path`` to ``<stem>.partN<suffix>`` if it exists.

    N is one greater than the highest ``partK`` already present in
    the same directory, so the CURRENT run always keeps the base name
    and the rotated chunks sort naturally: part1 is the oldest,
    partN is the most recent just-closed chunk.

    Returns the new rotated path, or ``None`` if the source did not
    exist (first run — nothing to rotate).
    """
    if not path.exists():
        return None
    stem = path.stem  # e.g. "training"
    suffix = path.suffix  # e.g. ".log"  (leading dot included)
    # Regex against siblings; ``re.escape`` protects against dots in the stem.
    pat = re.compile(rf"^{re.escape(stem)}\.part(\d+){re.escape(suffix)}$")
    used: set[int] = set()
    for sibling in path.parent.iterdir():
        m = pat.match(sibling.name)
        if m:
            used.add(int(m.group(1)))
    n = max(used, default=0) + 1
    rotated = path.with_name(f"{stem}.part{n}{suffix}")
    path.rename(rotated)
    return rotated


# ── Appendable history reload ─────────────────────────────────────────


def load_history(path: Path) -> list:
    """Load a list-of-dicts history file. Return ``[]`` if absent/empty.

    Used for ``training_history.json``, ``eval_history.json``, and
    ``eval_sample_history.json``. If the file exists but is not a list
    (unexpected schema drift), raise ``ValueError`` rather than
    silently returning ``[]`` — the caller should know that the
    history on disk does not match the expected shape.
    """
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(
            f"{path} has unexpected schema: expected a JSON list, "
            f"got {type(data).__name__}. Refusing to resume without "
            f"operator intervention."
        )
    return data


# ── SwanLab metadata (swanlab.json) ───────────────────────────────────


def _meta_path(log_dir: Path) -> Path:
    return log_dir / SWANLAB_META_FILENAME


def load_swanlab_meta(log_dir: Path) -> Optional[dict]:
    """Read ``logs/<exp>/swanlab.json``. Returns ``None`` if absent.

    Fail-fast on malformed JSON: we would rather abort than silently
    pretend there is no previous run id.
    """
    path = _meta_path(log_dir)
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    return json.loads(text)  # will raise on malformed JSON — intentional


def _utcnow_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def init_swanlab_meta(
    log_dir: Path,
    project: str,
    experiment_name: str,
    swanlab_id: str,
) -> Path:
    """Write a FRESH swanlab.json for a new run.

    Called exactly once at the start of a non-resume run. Overwrites
    any existing file — a fresh run is, by definition, starting over.
    Callers should not invoke this during resume; use
    ``record_resume_event`` instead.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    now = _utcnow_iso()
    meta = {
        "swanlab_id": swanlab_id,
        "project": project,
        "experiment_name": experiment_name,
        "created_at": now,
        "last_resumed_at": None,
        "resume_count": 0,
        "resume_events": [],
    }
    path = _meta_path(log_dir)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def record_resume_event(
    log_dir: Path,
    checkpoint_path: Path,
    epoch: int,
    global_step: int,
) -> Path:
    """Append a resume event and bump counters in swanlab.json.

    The file MUST already exist (produced by ``init_swanlab_meta``
    on the original run) and MUST contain a non-empty
    ``swanlab_id`` — otherwise the caller has no business resuming
    without passing ``--swanlab-id`` explicitly.
    """
    path = _meta_path(log_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot record resume event — {path} does not exist. "
            f"The original run did not register its SwanLab id on "
            f"disk. Re-run with --swanlab-id <id> to reconstruct it."
        )
    meta = json.loads(path.read_text(encoding="utf-8"))
    if not meta.get("swanlab_id"):
        raise ValueError(
            f"{path} has empty swanlab_id. Refusing to continue — "
            f"pass --swanlab-id <id> explicitly to repair the file."
        )
    now = _utcnow_iso()
    meta["last_resumed_at"] = now
    meta["resume_count"] = int(meta.get("resume_count", 0)) + 1
    meta.setdefault("resume_events", []).append(
        {
            "at": now,
            "from_checkpoint": str(checkpoint_path),
            "epoch": int(epoch),
            "global_step": int(global_step),
        }
    )
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


# ── SwanLab id resolution (the precedence chain) ──────────────────────


class SwanLabIdMissingError(RuntimeError):
    """Raised when resume is requested but no swanlab id can be found.

    A distinct exception type makes the failure mode easy to catch in
    tests and keeps the message formatted in exactly one place.
    """


def resolve_swanlab_id(
    cli_id: str,
    meta: Optional[dict],
    log_dir: Path,
    resume: bool,
) -> str:
    """Pick the SwanLab run id for this process.

    Precedence (highest first):
      1. Explicit ``--swanlab-id`` from the CLI.
      2. ``swanlab_id`` in ``logs/<exp>/swanlab.json``.
      3. Hard error — refuse to start a disconnected run.

    For a fresh (non-resume) run, returns the empty string: the
    caller then lets SwanLab allocate a new id and persists it via
    ``init_swanlab_meta``.
    """
    if not resume:
        return ""
    if cli_id:
        return cli_id
    if meta and meta.get("swanlab_id"):
        return str(meta["swanlab_id"])
    raise SwanLabIdMissingError(
        "--resume was set but no SwanLab id could be found.\n"
        f"  1. Pass --swanlab-id <id> explicitly, OR\n"
        f"  2. Ensure {_meta_path(log_dir)} exists from the original run.\n"
        "Refusing to start a disconnected SwanLab run."
    )
