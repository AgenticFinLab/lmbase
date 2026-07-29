#!/usr/bin/env python3
"""Pre-download HuggingFace models into a project-controlled local root.

Arguments (every flag must be explicit on the CLI -- no hidden defaults):
    --root ENV_KEY        REQUIRED. .env key name holding the local root
                          (e.g. LabServerModelPath -> /Data/HFModels).
    -m / --model REPO_ID  HF repo id (repeatable). Mutually exclusive with --all.
    --all                 Download every entry in the curated DEFAULT_MODELS registry.
    --list                Print size table of local snapshots under --root.
    --verify REPO_ID      Validate a local snapshot via AutoConfig.
    --include GLOB        Allow-only download filter (repeatable).
    --exclude GLOB        Ignore filter (repeatable; defaults skip non-PyTorch).
    --push PROFILE        OPTIONAL. After download, push models to a remote server.
                          PROFILE = .env server profile prefix for SSH connection
                          (resolves <Profile>{User,Host,Port} -- but NOT password).
    --push-root ENV_KEY    REQUIRED when --push is set. .env key holding the
                          REMOTE destination root (e.g. CampusServerModelPath).
    --push-server-password ENV_KEY
                          REQUIRED when --push is set. EXPLICIT .env key holding
                          the SSH password for the push target (e.g.
                          ``CampusServerPWD``). The program reads exactly this
                          .env key -- there is NO prefix-based derivation. Pass
                          an empty value (i.e. set the .env key to empty) to
                          force key-based SSH auth (no sshpass).
    -j / --jobs N         Parallel SCP workers for --push (default: 4).
    --dry-run             Print planned actions; touch no files.

Effects:
    Root path   : resolved from ./.env[<ENV_KEY>]
    HF token    : resolved from ./.env[HF_TOKEN] (optional; needed for gated repos only)
    Layout      : <root>/<org>/<name>/  (org prefix preserved verbatim)
    Completeness: ALL files (weights + tokenizer + configs) downloaded by default
                  so the snapshot is DIRECTLY USABLE (no further network fetch)
    Idempotent  : re-runs skip completed files (ETag HTTP HEAD only)
    Push (if --push):
      - SSH probes remote file inventory (path + size)
      - Skips files already on remote with matching size
      - Parallel SCP for missing files only (-j workers)
      - Avoids redundant/invalid transfers

Usage (every command is COPY-PASTE READY -- single-line)::

    python3 HFModelsCollect.py --root LabServerModelPath --all
    python3 HFModelsCollect.py --root LabServerModelPath --all --push CampusServer --push-root CampusServerModelPath --push-server-password CampusServerPWD
    python3 HFModelsCollect.py --root LabServerModelPath --all --push CampusServer --push-root CampusServerModelPath --push-server-password CampusServerPWD -j 8
    python3 HFModelsCollect.py --root LabServerModelPath --all --push CampusServer --push-root CampusServerModelPath --push-server-password CampusServerPWD --dry-run
    python3 HFModelsCollect.py --root CampusServerModelPath --all
    python3 HFModelsCollect.py --root LabServerModelPath --all --dry-run
    python3 HFModelsCollect.py --root LabServerModelPath -m Qwen/Qwen2.5-0.5B
    python3 HFModelsCollect.py --root LabServerModelPath -m Qwen/Qwen2.5-0.5B -m Qwen/Qwen3-1.7B
    python3 HFModelsCollect.py --root LabServerModelPath --list
    python3 HFModelsCollect.py --root LabServerModelPath --verify Qwen/Qwen2.5-0.5B
    python3 HFModelsCollect.py --root LabServerModelPath --all --include '*.safetensors' --include '*.json' --include 'tokenizer.model'
    python3 HFModelsCollect.py --root LabServerModelPath --all --exclude ''

----------------------------------------------------------------------
Setup  (do this ONCE per workstation)
----------------------------------------------------------------------
Step 1 -- add .env keys for each machine's model root::

        LabServerModelPath    = /Data/HFModels
        CampusServerModelPath = /data/user/sijiachen/HFModels

Step 2 -- (optional) add an HF access token for gated repos::

        HF_TOKEN = hf_xxx...

    Public repos (Qwen / sentence-transformers) need NO token.

Step 3 -- ensure ``huggingface_hub`` is importable::

        python3 -c "import huggingface_hub; print(huggingface_hub.__version__)"

----------------------------------------------------------------------
Downstream consumption (model is directly usable -- all files local)
----------------------------------------------------------------------
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HF_ROOT = os.environ["LabServerModelPath"]
    model = AutoModelForCausalLM.from_pretrained(f"{HF_ROOT}/Qwen/Qwen2.5-0.5B")
    tokenizer = AutoTokenizer.from_pretrained(f"{HF_ROOT}/Qwen/Qwen2.5-0.5B")

----------------------------------------------------------------------
Re-run safety / idempotency
----------------------------------------------------------------------
* Re-runs skip completed files (ETag HTTP HEAD only -- no re-download).
* Interrupted downloads resume cleanly on next invocation.
* Gated-repo 401/403 is caught PER MODEL (loop continues with next).
* Repository-not-found (typo) is caught per-model with a clear FAIL line.

----------------------------------------------------------------------
Troubleshooting
----------------------------------------------------------------------
* ``[FATAL] .env key ... is empty or not found`` -- add the key to
  ``./.env`` (e.g. ``LabServerModelPath = /Data/HFModels``).
* ``[FAIL] <repo> GATED`` -- visit ``https://huggingface.co/<repo>``,
  click "Request access", then set ``HF_TOKEN`` in ``./.env``.
* ``[FAIL] <repo> NOT FOUND`` -- typo in the repo id, OR private
  without token.
* ``429 Too Many Requests`` -- wait a few minutes and re-run.
* ``No space left on device`` -- use a different .env key pointing
  to a larger filesystem.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from huggingface_hub.utils import (
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
)
from transformers import AutoConfig

# --- .env loading ------------------------------------------------------
# Load key-value pairs from ``./.env`` at the repo root BEFORE any
# resolution below. We never hard-code a path or token in this source
# file -- everything sensitive flows through ``./.env``.
# ``override=False`` so a real env var (exported in the shell) wins.
load_dotenv(
    dotenv_path=Path(__file__).resolve().parent / ".env",
    override=False,
)

DEFAULT_HF_TOKEN = (os.environ.get("HF_TOKEN") or "").strip()


# Curated registry of the HF repos this project actually consumes.
# ALL Qwen2.5 + Qwen3 + Llama models under 40B active parameters.
# Order: smallest first so the operator gets quick feedback.
# NOTE: Meta Llama repos are GATED -- accept the license at
#   https://huggingface.co/meta-llama and set HF_TOKEN in .env.
DEFAULT_MODELS: list[str] = [
    # --- Qwen2.5 (base, dense) ---
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-14B",
    # "Qwen/Qwen2.5-32B",
    # --- Qwen3 (dense) ---
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    # "Qwen/Qwen3-32B",
    # --- Llama 3.2 (dense, text-only) ---
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    # --- Llama 3.1 (dense) ---
    "meta-llama/Llama-3.1-8B",
]

# Default ignore globs -- ONLY skip alternative weight formats that this
# project NEVER loads (gguf/ggml for llama.cpp, msgpack/h5/ot for
# Flax/TF/ONNX). Everything else -- PyTorch weights (*.bin,
# *.safetensors), ALL tokenizer files, ALL config JSONs -- is
# downloaded so the local snapshot is DIRECTLY USABLE without any
# further network fetch.
DEFAULT_IGNORE_PATTERNS: list[str] = [
    "*.gguf",
    "*.ggml",
    "*.msgpack",
    "*.h5",
    "*.ot",
    "flax_model.*",
    "tf_model.*",
]

# Standard rule width for headers / separators.
LINE_W = 78
# Fixed width of the [TAG ] column.
TAG_W = 6
# Lock for atomic stdout output from parallel push workers.
_OUTPUT_LOCK = threading.Lock()

# Profile attributes consumed by --push (SSH connection only).
# NOTE: Password is NOT in this list -- it is resolved EXPLICITLY via
# the --push-server-password ENV_KEY CLI flag. Prefix-based password
# derivation is intentionally forbidden so that the operator must name
# the .env key for the password directly.
_PROFILE_ATTRS = ("IP", "User", "Host", "Port")


# ======================================================================
# Display helpers
# ======================================================================
def _tag(tag: str) -> str:
    """Right-pad a status tag to ``TAG_W`` for column-aligned output."""
    return f"[{tag.upper():<{TAG_W - 2}}]"


def log(tag: str, msg: str, *, indent: int = 0) -> None:
    """Emit a single ``[TAG ]  msg`` line with optional left indent."""
    pad = "  " * indent
    print(f"{_tag(tag)} {pad}{msg}")


def hr(char: str = "=") -> None:
    """Print a horizontal rule spanning ``LINE_W`` chars."""
    print(char * LINE_W)


def header(title: str, char: str = "=") -> None:
    """Print a banner: rule, centred title, rule."""
    hr(char)
    print(title.center(LINE_W))
    hr(char)


def human_size(n_bytes: int) -> str:
    """Format a byte count as the largest sensible unit (B / KB / .. / TB)."""
    size = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0:
            return f"{size:7.2f} {unit}"
        size /= 1024.0
    return f"{size:7.2f} PB"


def dir_size(path: Path) -> int:
    """Return the total byte size of all regular files under ``path``."""
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


# ======================================================================
# CLI
# ======================================================================
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for model download and optional push."""
    parser = argparse.ArgumentParser(
        description=(
            "Pre-download HuggingFace model snapshots into a local root "
            "resolved from ``./.env`` via the key name passed to --root "
            "(e.g. ``--root LabServerModelPath``)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root",
        required=True,
        metavar="ENV_KEY",
        help=(
            "Name of the .env key holding the local destination root "
            "(e.g. ``LabServerModelPath``). Code resolves it via "
            "``os.environ[ENV_KEY]``. REQUIRED."
        ),
    )
    parser.add_argument(
        "-m",
        "--model",
        action="append",
        default=[],
        metavar="REPO_ID",
        help=(
            "HuggingFace repo id like ``Qwen/Qwen2.5-0.5B``. May be "
            "passed multiple times. Mutually exclusive with --all."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Download every entry in DEFAULT_MODELS (the curated "
            f"registry, currently {len(DEFAULT_MODELS)} models)."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List snapshots already present under --root, with sizes.",
    )
    parser.add_argument(
        "--verify",
        metavar="REPO_ID",
        help=(
            "Run ``AutoConfig.from_pretrained`` on the local snapshot "
            "to confirm it is structurally valid. Reads config.json "
            "only -- no weight load, no GPU."
        ),
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="GLOB",
        help=(
            "Allow-only glob (repeatable). When set, ONLY matching "
            "files are fetched (e.g. --include '*.safetensors')."
        ),
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=list(DEFAULT_IGNORE_PATTERNS),
        metavar="GLOB",
        help=(
            "Ignore glob (repeatable). Defaults skip non-PyTorch "
            "formats (gguf / msgpack / h5 / ot / flax_model.* / "
            "tf_model.*). Pass an empty --exclude to clear."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions; touch no files.",
    )
    parser.add_argument(
        "--push",
        metavar="PROFILE",
        help=(
            "After local download, push models to a remote server. "
            "PROFILE identifies the .env server profile prefix for SSH "
            "connection (resolves <Profile>{User,Host,Port}). The SSH "
            "password is NOT derived from this prefix -- it must be "
            "named EXPLICITLY via --push-server-password ENV_KEY. "
            "Must be paired with --push-root and "
            "--push-server-password. OPTIONAL."
        ),
    )
    parser.add_argument(
        "--push-root",
        metavar="ENV_KEY",
        help=(
            "Name of the .env key holding the REMOTE destination root "
            "(e.g. ``CampusServerModelPath``). Code resolves it via "
            "``os.environ[ENV_KEY]``. REQUIRED when --push is set."
        ),
    )
    parser.add_argument(
        "--push-server-password",
        metavar="ENV_KEY",
        help=(
            "EXPLICIT name of the .env key holding the SSH password "
            "for the --push target (e.g. ``CampusServerPWD``). The "
            "program reads exactly this .env key via "
            "``os.environ[ENV_KEY]`` -- there is NO prefix-based "
            "derivation from --push. If the named .env key is set to "
            "an empty value, sshpass is NOT used (key-based SSH auth "
            "is assumed). REQUIRED when --push is set."
        ),
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=4,
        metavar="N",
        help="Parallel SCP workers for --push (default: 4).",
    )
    return parser.parse_args()


# ======================================================================
# Core operations
# ======================================================================
def resolve_local_dir(root: Path, repo_id: str) -> Path:
    """Map ``org/name`` -> ``<root>/org/name`` (org prefix preserved).

    Bare repo ids without a slash (rare) land directly under ``root``.
    """
    parts = repo_id.split("/")
    return root.joinpath(*parts)


def _count_files(path: Path) -> int:
    """Count regular files under *path*."""
    if not path.is_dir():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file())


def _has_config(local_dir: Path) -> bool:
    """True if ``config.json`` exists with non-zero size."""
    f = local_dir / "config.json"
    return f.is_file() and f.stat().st_size > 0


def _has_tokenizer(local_dir: Path) -> bool:
    """True if ``tokenizer_config.json`` exists with non-zero size."""
    f = local_dir / "tokenizer_config.json"
    return f.is_file() and f.stat().st_size > 0


def _weights_complete(local_dir: Path) -> tuple[bool, str]:
    """Rigorously verify ALL weight files for a model are present.

    Returns (is_complete, diagnostic_reason_string).

    CRITICAL: This function must NOT give a false-positive for sharded
    models where only some shards were downloaded. It verifies
    completeness by parsing the index file and checking every shard.

    Logic (tried in order, first match wins):
      1. ``model.safetensors.index.json`` exists (sharded safetensors)
         -> parse weight_map, verify EVERY referenced shard file exists
            on disk with non-zero size.
      2. ``pytorch_model.bin.index.json`` exists (sharded bin)
         -> same verification for .bin shards.
      3. ``model.safetensors`` exists (single-file safetensors)
         -> verify non-zero size.
      4. ``pytorch_model.bin`` exists (single-file bin)
         -> verify non-zero size.
      5. None of the above -> incomplete.
    """
    # --- Case 1: Sharded safetensors (most common for modern models) -------
    st_index = local_dir / "model.safetensors.index.json"
    if st_index.is_file():
        try:
            data = json.loads(st_index.read_text(encoding="utf-8"))
            # weight_map: {param_name: shard_filename}
            shard_files = sorted(set(data["weight_map"].values()))
        except (json.JSONDecodeError, KeyError, TypeError):
            return False, "safetensors index.json corrupt or unreadable"
        missing = []
        for shard in shard_files:
            shard_path = local_dir / shard
            if not shard_path.is_file() or shard_path.stat().st_size == 0:
                missing.append(shard)
        if missing:
            return (
                False,
                f"{len(missing)}/{len(shard_files)} safetensors shard(s) MISSING",
            )
        return True, f"all {len(shard_files)} safetensors shards present"

    # --- Case 2: Sharded pytorch_model.bin (older format) -----------------
    bin_index = local_dir / "pytorch_model.bin.index.json"
    if bin_index.is_file():
        try:
            data = json.loads(bin_index.read_text(encoding="utf-8"))
            shard_files = sorted(set(data["weight_map"].values()))
        except (json.JSONDecodeError, KeyError, TypeError):
            return False, "bin index.json corrupt or unreadable"
        missing = []
        for shard in shard_files:
            shard_path = local_dir / shard
            if not shard_path.is_file() or shard_path.stat().st_size == 0:
                missing.append(shard)
        if missing:
            return (
                False,
                f"{len(missing)}/{len(shard_files)} bin shard(s) MISSING",
            )
        return True, f"all {len(shard_files)} bin shards present"

    # --- Case 3: Single-file safetensors ----------------------------------
    single_st = local_dir / "model.safetensors"
    if single_st.is_file() and single_st.stat().st_size > 0:
        return True, "single model.safetensors present"

    # --- Case 4: Single-file pytorch_model.bin ----------------------------
    single_bin = local_dir / "pytorch_model.bin"
    if single_bin.is_file() and single_bin.stat().st_size > 0:
        return True, "single pytorch_model.bin present"

    # --- Case 5: No recognizable weight structure -------------------------
    # Check if any weight-like files exist without a proper index
    any_st = list(local_dir.glob("*.safetensors"))
    any_bin = list(local_dir.glob("*.bin"))
    if any_st or any_bin:
        return (
            False,
            f"weight files exist ({len(any_st)} st + {len(any_bin)} bin) "
            f"but no index file to verify completeness",
        )

    return False, "no weight files found"


def download_one(
    *,
    repo_id: str,
    root: Path,
    token: str,
    allow_patterns: list[str],
    ignore_patterns: list[str],
    dry_run: bool,
) -> bool:
    """Download a single repo into ``<root>/<repo_id>``.

    Pre-checks local state with RIGOROUS completeness verification:
      [SKIP]   - model VERIFIED complete (config + ALL weight shards via
                 index.json + tokenizer). NO network call -- truly skipped.
      [RESUME] - partial/incomplete detected (shard check failed) ->
                 calls snapshot_download to fetch remaining files.
      [XFR]    - fresh download (directory empty or missing).
      [OK]     - download success (with size and file count).
      [FAIL]   - recoverable error (gated / 404 / network).

    SAFETY GUARANTEE:
      [SKIP] is ONLY emitted when ALL pass:
        1. config.json exists (non-zero size)
        2. tokenizer_config.json exists (non-zero size)
        3. _weights_complete() returns True (for sharded models this
           parses index.json and verifies EVERY shard file exists with
           non-zero size -- a model with 1/8 shards will NEVER be skipped)
      If ANY check fails -> snapshot_download is called (never skipped).

    Returns True on success (incl. verified skip), False on error.
    """
    local_dir = resolve_local_dir(root, repo_id)

    # --- Rigorous local completeness check --------------------------------
    has_cfg = _has_config(local_dir)
    has_tok = _has_tokenizer(local_dir)
    weights_ok, weights_reason = _weights_complete(local_dir)

    if has_cfg and has_tok and weights_ok:
        # Rigorously verified: config + tokenizer + ALL shards present.
        # Truly skip -- no network call, no progress bars.
        file_count = _count_files(local_dir)
        size = dir_size(local_dir)
        log(
            "SKIP",
            f"{repo_id}  VERIFIED complete ({human_size(size).strip()}, "
            f"{file_count} files, {weights_reason})",
        )
        return True

    # --- Not complete: show WHY it failed the completeness check -----------
    if has_cfg or local_dir.is_dir():
        log(
            "RESUME",
            f"{repo_id}  incomplete -> downloading "
            f"(cfg={'OK' if has_cfg else 'MISSING'}, "
            f"tok={'OK' if has_tok else 'MISSING'}, "
            f"wt={weights_reason})",
        )
    else:
        log("XFR", f"{repo_id}  ->  {local_dir}")

    if dry_run:
        log("DRY", "snapshot_download skipped (--dry-run)", indent=1)
        return True

    # --- Download (handles partial resume + fresh) -------------------------
    files_before = _count_files(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            token=token or None,
            allow_patterns=allow_patterns or None,
            ignore_patterns=ignore_patterns or None,
        )
    except GatedRepoError as exc:
        log(
            "FAIL",
            f"{repo_id}  GATED -- visit https://huggingface.co/{repo_id} "
            f"to request access, then re-run with HF_TOKEN set "
            f"({exc.__class__.__name__})",
            indent=1,
        )
        return False
    except RepositoryNotFoundError:
        log(
            "FAIL",
            f"{repo_id}  NOT FOUND (typo? private without token?)",
            indent=1,
        )
        return False
    except (HfHubHTTPError, LocalEntryNotFoundError, OSError) as exc:
        log("FAIL", f"{repo_id}  {exc.__class__.__name__}: {exc}", indent=1)
        return False

    # --- Post-download verification ----------------------------------------
    files_after = _count_files(local_dir)
    new_files = files_after - files_before
    size = dir_size(local_dir)

    # Re-run completeness check after download to confirm integrity
    post_wt_ok, post_wt_reason = _weights_complete(local_dir)
    post_cfg = _has_config(local_dir)
    post_tok = _has_tokenizer(local_dir)

    if not (post_cfg and post_tok and post_wt_ok):
        log(
            "WARN",
            f"{repo_id}  download finished but NOT fully complete: "
            f"cfg={'OK' if post_cfg else 'MISSING'}, "
            f"tok={'OK' if post_tok else 'MISSING'}, "
            f"wt={post_wt_reason}. "
            f"Re-run to retry.",
            indent=1,
        )

    if new_files > 0:
        log(
            "OK",
            f"{repo_id}  +{new_files} new file(s), "
            f"total {human_size(size).strip()} ({post_wt_reason})",
            indent=1,
        )
    else:
        log(
            "OK",
            f"{repo_id}  up-to-date ({human_size(size).strip()}, "
            f"{files_after} files)",
            indent=1,
        )
    return True


def verify_one(*, repo_id: str, root: Path) -> bool:
    """Confirm a local snapshot loads via ``AutoConfig.from_pretrained``.

    No weight materialisation -- this just parses ``config.json`` and
    validates the architecture is registered in ``transformers``.
    """
    local_dir = resolve_local_dir(root, repo_id)
    if not local_dir.is_dir():
        log("FAIL", f"{repo_id}  no local snapshot at {local_dir}")
        return False
    try:
        cfg = AutoConfig.from_pretrained(str(local_dir))
    except (OSError, ValueError) as exc:
        log("FAIL", f"{repo_id}  AutoConfig failed: {exc}")
        return False
    log(
        "OK",
        f"{repo_id}  model_type={cfg.model_type}  "
        f"hidden_size={getattr(cfg, 'hidden_size', '?')}  "
        f"vocab_size={getattr(cfg, 'vocab_size', '?')}",
    )
    return True


def list_local(root: Path) -> None:
    """Print a size table of every snapshot directly under ``<root>``.

    Walks two levels (``org/name``) and one level (``name`` for repos
    without a slash) so the listing mirrors :func:`resolve_local_dir`.
    """
    if not root.is_dir():
        log("WARN", f"root does not exist yet: {root}")
        return
    rows: list[tuple[str, str, Path]] = []
    for org_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        children = sorted(p for p in org_dir.iterdir() if p.is_dir())
        if not children:
            rows.append((org_dir.name, human_size(dir_size(org_dir)), org_dir))
            continue
        for child in children:
            repo_id = f"{org_dir.name}/{child.name}"
            rows.append((repo_id, human_size(dir_size(child)), child))
    if not rows:
        log("INFO", f"no snapshots present under {root}")
        return
    name_w = max(len(r[0]) for r in rows)
    header(f"Local HF snapshots under {root}", char="-")
    for repo_id, size, path in rows:
        print(f"  {repo_id:<{name_w}}  {size}  {path}")
    hr("-")
    total = sum(dir_size(r[2]) for r in rows)
    log("INFO", f"{len(rows)} snapshot(s), total {human_size(total).strip()}")


# ======================================================================
# Push to remote server (--push PROFILE)
# ======================================================================
def load_push_profile(name: str) -> dict[str, str]:
    """Resolve a named server profile for push SSH connection.

    Reads ``<name><Attr>`` from environment for each attribute in
    ``_PROFILE_ATTRS`` (currently: IP, User, Host, Port). ``Host``
    falls back to ``IP``. Raises SystemExit when profile is empty or
    User/Host missing.

    NOTE: The SSH password is INTENTIONALLY NOT resolved here. It is
    handled separately via the explicit ``--push-server-password
    ENV_KEY`` CLI flag so the operator names the password .env key
    directly. See :func:`resolve_push_password`.
    """
    if not name:
        raise SystemExit("[FATAL] --push requires a profile name.")
    out = {a: (os.environ.get(f"{name}{a}") or "").strip() for a in _PROFILE_ATTRS}
    if not any(out.values()):
        raise SystemExit(
            f"[FATAL] push profile {name!r} not found in ./.env -- "
            f"expected at least one of: "
            + ", ".join(f"{name}{a}" for a in _PROFILE_ATTRS)
        )
    if not out["Host"]:
        out["Host"] = out["IP"]
    if not out["User"] or not out["Host"]:
        raise SystemExit(
            f"[FATAL] push profile {name!r} missing User or Host/IP. "
            f"Set {name}User and {name}Host (or {name}IP) in ./.env."
        )
    return out


def _ssh_port_flag(port: str) -> str:
    """Return `` -p N`` or empty string."""
    return f" -p {port}" if port else ""


def resolve_push_password(env_key: str) -> str:
    """Resolve the SSH push password from an EXPLICITLY-named .env key.

    The operator passes ``--push-server-password CampusServerPWD`` (or
    similar). This function reads ``os.environ[env_key]`` directly --
    there is NO prefix-based derivation from the --push profile name.

    Empty value semantics:
      - Empty string (or env key absent): caller treats this as
        "no sshpass; rely on key-based SSH auth". This is a VALID
        configuration -- not a fatal error.
      - Non-empty string: used as the SSH password via ``sshpass -p``.

    Raises SystemExit ONLY when ``env_key`` itself is empty (i.e. the
    operator forgot to pass the flag).
    """
    if not env_key:
        raise SystemExit(
            "[FATAL] --push-server-password is required when --push is set. "
            "Pass the .env key name holding the SSH password "
            "(e.g. --push-server-password CampusServerPWD)."
        )
    # Read the named .env key. Empty value is allowed and means
    # "no password -> key-based SSH auth".
    return (os.environ.get(env_key) or "").strip()


def _scp_port_flag(port: str) -> str:
    """Return `` -P N`` or empty string (SCP uses uppercase P)."""
    return f" -P {port}" if port else ""


def _auth_prefix(password: str) -> str:
    """Return ``sshpass -p '...' `` prefix, or empty string."""
    if not password:
        return ""
    escaped = password.replace("'", "'\\''")
    return f"sshpass -p '{escaped}' "


def _remote_file_inventory(
    *, auth: str, user_host: str, port: str, remote_dir: str
) -> tuple[dict[str, int], bool]:
    """SSH into remote and list files with sizes under ``remote_dir``.

    Returns (inventory_dict, success_flag):
      - ({relative_path: size_bytes}, True)  -- probed successfully
      - ({}, True)                           -- dir does not exist (empty)
      - ({}, False)                          -- SSH/stat command FAILED

    The caller MUST check ``success_flag``. When False, the inventory is
    UNRELIABLE and the caller must NOT assume all files are missing.
    Proceeding with an unreliable empty inventory would cause full
    redundant re-push of files already present on remote.
    """
    cmd = (
        f"{auth}ssh{_ssh_port_flag(port)} {user_host} "
        f"\"if [ -d '{remote_dir}' ]; then "
        f"find '{remote_dir}' -type f -exec stat --format='%n %s' {{}} \\;; "
        f"else echo '__EMPTY__'; fi\""
    )
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        # Try macOS stat format as fallback
        cmd_mac = (
            f"{auth}ssh{_ssh_port_flag(port)} {user_host} "
            f"\"if [ -d '{remote_dir}' ]; then "
            f"find '{remote_dir}' -type f | while read f; do "
            f'printf \'%s %s\\n\' \\"\\$f\\" \\$(stat -f%z \\"\\$f\\"); done; '
            f"else echo '__EMPTY__'; fi\""
        )
        proc = subprocess.run(
            cmd_mac, shell=True, capture_output=True, text=True, check=False
        )
    if proc.returncode != 0:
        log(
            "FAIL",
            f"SSH stat probe failed (rc={proc.returncode}): " f"{proc.stderr.strip()}",
        )
        return {}, False
    result: dict[str, int] = {}
    prefix = remote_dir.rstrip("/") + "/"
    for line in proc.stdout.strip().splitlines():
        if line == "__EMPTY__":
            # Remote directory does not exist -- all files are genuinely new
            return {}, True
        # Format: /full/path/to/file SIZE
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        fpath, size_str = parts
        try:
            size = int(size_str)
        except ValueError:
            continue
        if fpath.startswith(prefix):
            rel = fpath[len(prefix) :]
            result[rel] = size
    return result, True


def _local_file_inventory(local_dir: Path) -> dict[str, int]:
    """List all files under ``local_dir`` with their sizes.

    Returns ``{relative_path: size_bytes}``.
    Excludes internal HF cache metadata (``.huggingface/``) that is
    not needed on the remote and would be redundant.
    """
    if not local_dir.is_dir():
        return {}
    result: dict[str, int] = {}
    for p in local_dir.rglob("*"):
        if p.is_file():
            rel = str(p.relative_to(local_dir))
            # Skip HF internal cache dirs (download metadata, not model content)
            if rel.startswith(".huggingface") or rel.startswith(".cache"):
                continue
            result[rel] = p.stat().st_size
    return result


def _push_one_file(
    *,
    local_path: str,
    remote_path: str,
    auth: str,
    user_host: str,
    port: str,
    dry_run: bool,
) -> tuple[str, bool]:
    """SCP a single file to remote. Returns (relative_path, success)."""
    # Extract base filename for log messages
    rel = Path(local_path).name
    if dry_run:
        return rel, True
    # Ensure remote parent dir exists
    remote_parent = str(Path(remote_path).parent)
    mkdir_cmd = (
        f"{auth}ssh{_ssh_port_flag(port)} {user_host} "
        f"\"mkdir -p '{remote_parent}'\""
    )
    subprocess.run(mkdir_cmd, shell=True, capture_output=True, check=False)
    # SCP the file
    scp_cmd = (
        f"{auth}scp{_scp_port_flag(port)} '{local_path}' "
        f"{user_host}:'{remote_path}'"
    )
    proc = subprocess.run(
        scp_cmd, shell=True, capture_output=True, text=True, check=False
    )
    return rel, (proc.returncode == 0)


def _check_ssh_connectivity(*, auth: str, user_host: str, port: str) -> bool:
    """Verify SSH connectivity before any push operation.

    Runs ``ssh <host> echo ok`` and checks for success.
    Returns True if connection works, False otherwise.
    """
    cmd = f"{auth}ssh{_ssh_port_flag(port)} {user_host} 'echo __SSH_OK__'"
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
    return proc.returncode == 0 and "__SSH_OK__" in proc.stdout


def push_models_to_server(
    *,
    repos: list[str],
    local_root: Path,
    profile: dict[str, str],
    password: str,
    remote_root: str,
    jobs: int,
    dry_run: bool,
) -> tuple[int, int, int]:
    """Push locally-present models to a remote server.

    Pipeline per model:
      Phase 1 [CHECK]  : SSH connectivity pre-flight
      Phase 2 [PROBE]  : Inventory remote files (path + size)
      Phase 3 [DIFF]   : Compare local vs remote, compute delta
      Phase 4 [MKDIR]  : Batch-create remote parent directories
      Phase 5 [PUSH]   : Parallel SCP for missing files
      Phase 6 [VERIFY] : Re-probe remote to confirm arrival

    Authentication:
      ``password`` is supplied EXPLICITLY by the caller (resolved from
      the .env key named via --push-server-password). Empty string
      means "no sshpass; key-based SSH auth".

    Returns (total_files_pushed, total_files_skipped, total_failed).
    """
    user_host = f"{profile['User']}@{profile['Host']}"
    port = profile["Port"]
    auth = _auth_prefix(password)
    remote_root = remote_root.rstrip("/")

    # --- Phase 1: SSH connectivity check ---------------------------------
    log("CHECK", f"testing SSH to {user_host} ...")
    if not dry_run and not _check_ssh_connectivity(
        auth=auth, user_host=user_host, port=port
    ):
        log(
            "FAIL",
            f"cannot connect to {user_host} -- check profile, "
            f"password, network. Aborting push.",
        )
        return 0, 0, 0
    log("OK", f"SSH connection to {user_host} verified")
    print()

    total_pushed = 0
    total_skipped = 0
    total_failed = 0
    total_bytes = 0

    for repo_id in repos:
        local_dir = resolve_local_dir(local_root, repo_id)
        if not local_dir.is_dir():
            log("SKIP", f"{repo_id}  not present locally -- nothing to push")
            continue

        remote_dir = f"{remote_root}/{repo_id}"

        # --- Phase 2: Probe remote inventory -----------------------------
        log("PROBE", f"{repo_id}  inventorying remote {user_host}:{remote_dir}")
        if dry_run:
            remote_inv: dict[str, int] = {}
            log("DRY", "remote probe skipped (--dry-run)", indent=1)
        else:
            remote_inv, probe_ok = _remote_file_inventory(
                auth=auth, user_host=user_host, port=port, remote_dir=remote_dir
            )
            if not probe_ok:
                # SAFETY: Cannot determine what's already on remote.
                # Proceeding would risk full redundant re-push.
                log(
                    "FAIL",
                    f"{repo_id}  remote inventory probe FAILED -- "
                    f"skipping this model to avoid redundant re-push. "
                    f"Fix SSH/stat issue and retry.",
                    indent=1,
                )
                total_failed += 1
                continue
            log("INFO", f"remote has {len(remote_inv)} file(s)", indent=1)

        # --- Phase 3: Diff local vs remote -------------------------------
        local_inv = _local_file_inventory(local_dir)
        local_total_size = sum(local_inv.values())
        log(
            "INFO",
            f"local has {len(local_inv)} file(s), "
            f"{human_size(local_total_size).strip()}",
            indent=1,
        )

        # Categorize each file: skip (size-matched) / new / size-changed
        # on remote with same size
        skip_matched: list[str] = []
        # not on remote at all
        new_files: list[str] = []
        # on remote but size differs
        changed_files: list[str] = []
        # zero-byte local file (corrupt / placeholder)
        skip_empty: list[str] = []

        for rel, local_size in sorted(local_inv.items()):
            if local_size == 0:
                skip_empty.append(rel)
                continue
            remote_size = remote_inv.get(rel)
            if remote_size is None:
                new_files.append(rel)
            elif remote_size == local_size:
                skip_matched.append(rel)
            else:
                changed_files.append(rel)

        missing = new_files + changed_files
        missing_bytes = sum(local_inv[r] for r in missing)
        skipped = len(skip_matched) + len(skip_empty)
        total_skipped += skipped

        # --- Skip reason breakdown (always shown) ---
        log("INFO", f"skip breakdown:", indent=1)
        log(
            "INFO",
            f"  size-matched (identical on remote): {len(skip_matched)}",
            indent=1,
        )
        if skip_empty:
            log("INFO", f"  zero-byte (not pushed): {len(skip_empty)}", indent=1)
        if new_files:
            log("INFO", f"  new (not on remote): {len(new_files)}", indent=1)
        if changed_files:
            log(
                "INFO",
                f"  size-changed (will overwrite): {len(changed_files)}",
                indent=1,
            )

        if not missing:
            log(
                "SKIP",
                f"{repo_id}  remote is FULLY up-to-date "
                f"({len(skip_matched)} files, all sizes match)",
            )
            continue

        log(
            "DIFF",
            f"{repo_id}  {len(missing)} file(s) to transfer "
            f"({human_size(missing_bytes).strip()}), "
            f"{skipped} skipped",
        )

        if dry_run:
            for rel in missing[:5]:
                sz = local_inv[rel]
                log("DRY", f"would push: {rel} ({human_size(sz).strip()})", indent=1)
            if len(missing) > 5:
                log("DRY", f"... and {len(missing) - 5} more", indent=1)
            total_pushed += len(missing)
            total_bytes += missing_bytes
            continue

        # --- Phase 4: Batch mkdir remote parent dirs ---------------------
        parents = sorted({str(Path(remote_dir) / Path(r).parent) for r in missing})
        mk_args = " ".join(f"'{p}'" for p in parents)
        mkdir_cmd = (
            f"{auth}ssh{_ssh_port_flag(port)} {user_host} " f'"mkdir -p {mk_args}"'
        )
        log("MKDIR", f"creating {len(parents)} remote dir(s)", indent=1)
        rc = subprocess.run(
            mkdir_cmd, shell=True, capture_output=True, text=True, check=False
        )
        if rc.returncode != 0:
            log(
                "FAIL",
                f"mkdir failed (rc={rc.returncode}): " f"{rc.stderr.strip()}",
                indent=1,
            )
            total_failed += len(missing)
            continue
        log("OK", "remote dirs ready", indent=1)

        # --- Phase 5: Parallel SCP ---------------------------------------
        push_items = [
            {
                "local": str(local_dir / rel),
                "remote": f"{remote_dir}/{rel}",
                "rel": rel,
                "size": local_inv[rel],
            }
            for rel in missing
        ]

        repo_ok = 0
        repo_fail = 0
        repo_bytes = 0
        effective_jobs = min(jobs, len(push_items))
        log(
            "PUSH",
            f"transferring {len(push_items)} file(s) with "
            f"{effective_jobs} parallel worker(s)",
            indent=1,
        )

        with ThreadPoolExecutor(max_workers=effective_jobs) as pool:
            futures = {
                pool.submit(
                    _push_one_file,
                    local_path=item["local"],
                    remote_path=item["remote"],
                    auth=auth,
                    user_host=user_host,
                    port=port,
                    dry_run=False,
                ): item
                for item in push_items
            }
            done_count = 0
            for fut in as_completed(futures):
                item = futures[fut]
                done_count += 1
                try:
                    _, ok = fut.result()
                except Exception as exc:
                    ok = False
                    with _OUTPUT_LOCK:
                        log(
                            "FAIL",
                            f"[{done_count}/{len(push_items)}] "
                            f"{item['rel']} "
                            f"({human_size(item['size']).strip()})  "
                            f"exception: {exc}",
                            indent=2,
                        )
                    continue
                with _OUTPUT_LOCK:
                    if ok:
                        repo_ok += 1
                        repo_bytes += item["size"]
                        log(
                            "XFR",
                            f"[{done_count}/{len(push_items)}] "
                            f"{item['rel']} "
                            f"({human_size(item['size']).strip()})",
                            indent=2,
                        )
                    else:
                        repo_fail += 1
                        log(
                            "FAIL",
                            f"[{done_count}/{len(push_items)}] "
                            f"{item['rel']} "
                            f"({human_size(item['size']).strip()})  "
                            f"scp returned non-zero",
                            indent=2,
                        )

        total_pushed += repo_ok
        total_failed += repo_fail
        total_bytes += repo_bytes

        # --- Phase 6: Post-push verification ------------------------------
        if repo_ok > 0 and repo_fail == 0:
            log("VERIFY", f"{repo_id}  re-probing remote ...", indent=1)
            verify_inv, verify_probe_ok = _remote_file_inventory(
                auth=auth, user_host=user_host, port=port, remote_dir=remote_dir
            )
            if not verify_probe_ok:
                log(
                    "WARN",
                    f"{repo_id}  verification probe FAILED -- "
                    f"cannot confirm files arrived. Manual check needed.",
                    indent=1,
                )
            else:
                verified = 0
                verify_fail = 0
                for rel in missing:
                    expected_size = local_inv[rel]
                    actual_size = verify_inv.get(rel)
                    if actual_size == expected_size:
                        verified += 1
                    else:
                        verify_fail += 1
                        log(
                            "FAIL",
                            f"verification mismatch: {rel} "
                            f"(expected {expected_size}, got {actual_size})",
                            indent=2,
                        )
                if verify_fail == 0:
                    log("OK", f"{repo_id}  all {verified} file(s) verified on remote")
                else:
                    log(
                        "WARN",
                        f"{repo_id}  {verify_fail}/{len(missing)} "
                        f"file(s) failed verification!",
                    )
                    total_failed += verify_fail
                    total_pushed -= verify_fail
        elif repo_fail == 0:
            log(
                "OK",
                f"{repo_id}  pushed {repo_ok}/{len(missing)} "
                f"({human_size(repo_bytes).strip()})",
            )
        else:
            log(
                "WARN",
                f"{repo_id}  pushed {repo_ok}/{len(missing)}, " f"{repo_fail} FAILED",
            )

    # --- Final summary line ----------------------------------------------
    if total_bytes > 0 or total_pushed > 0:
        log("INFO", f"total transferred: {human_size(total_bytes).strip()}")

    return total_pushed, total_skipped, total_failed


# ======================================================================
# Main
# ======================================================================
def resolve_root(env_key: str) -> Path:
    """Resolve the local destination root from a ``./.env`` key name.

    Reads ``os.environ[env_key]`` and returns an absolute Path. Raises
    ``SystemExit`` with an actionable message when the key is empty or
    missing from ``./.env``.
    """
    val = (os.environ.get(env_key) or "").strip()
    if not val:
        raise SystemExit(
            f"[FATAL] .env key {env_key!r} is empty or not found. "
            f"Add ``{env_key} = /path`` to ./.env and retry."
        )
    return Path(val).expanduser().resolve()


def main() -> int:
    """Entry point: download models and optionally push to remote."""
    args = parse_args()
    root = resolve_root(args.root)
    token = DEFAULT_HF_TOKEN

    if args.list:
        list_local(root)
        return 0

    if args.verify:
        ok = verify_one(repo_id=args.verify, root=root)
        return 0 if ok else 1

    if args.all and args.model:
        print(
            "[FATAL] --all and --model are mutually exclusive; pick one.",
            file=sys.stderr,
        )
        return 2
    if args.all:
        repos = list(DEFAULT_MODELS)
    elif args.model:
        repos = list(args.model)
    else:
        print(
            "[FATAL] nothing to do: pass --all, one or more --model REPO, "
            "--list, or --verify REPO. (Use --help for full usage.)",
            file=sys.stderr,
        )
        return 2

    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)

    include = list(args.include)
    exclude = [g for g in args.exclude if g]

    header("HFModelsCollect  --  pre-download HF snapshots to a local root")
    log("INFO", f"root key        : {args.root}")
    log("INFO", f"root path       : {root}")
    log("INFO", f"HF_TOKEN        : {'<set>' if token else '<unset>'}")
    log("INFO", f"allow_patterns  : {include or '<unset>'}")
    log("INFO", f"ignore_patterns : {exclude or '<unset>'}")
    log("INFO", f"models ({len(repos)}):")
    for repo in repos:
        log("INFO", repo, indent=2)
    if args.dry_run:
        log("INFO", "dry-run -- no files will be written")
    print()

    succeeded: list[str] = []
    failed: list[str] = []
    for repo in repos:
        ok = download_one(
            repo_id=repo,
            root=root,
            token=token,
            allow_patterns=include,
            ignore_patterns=exclude,
            dry_run=args.dry_run,
        )
        (succeeded if ok else failed).append(repo)

    print()
    header("Download Summary", char="-")
    log("INFO", f"succeeded : {len(succeeded)} / {len(repos)}")
    for repo in succeeded:
        log("OK", repo, indent=1)
    if failed:
        log("INFO", f"failed    : {len(failed)} / {len(repos)}")
        for repo in failed:
            log("FAIL", repo, indent=1)

    # ---- Push phase (optional) ------------------------------------------
    if args.push and succeeded:
        if not args.push_root:
            raise SystemExit(
                "[FATAL] --push requires --push-root ENV_KEY "
                "(e.g. --push-root CampusServerModelPath)."
            )
        if not args.push_server_password:
            raise SystemExit(
                "[FATAL] --push requires --push-server-password ENV_KEY "
                "(e.g. --push-server-password CampusServerPWD). "
                "This argument names the .env key holding the SSH "
                "password EXPLICITLY -- prefix-based derivation is "
                "intentionally not supported."
            )
        # Resolve remote root from .env key
        push_remote = (os.environ.get(args.push_root) or "").strip()
        if not push_remote:
            raise SystemExit(
                f"[FATAL] .env key {args.push_root!r} is empty or not found. "
                f"Add ``{args.push_root} = /path`` to ./.env and retry."
            )
        # Resolve SSH password from the EXPLICITLY-named .env key.
        # Empty value is allowed (means key-based SSH auth).
        push_password = resolve_push_password(args.push_server_password)

        print()
        push_profile = load_push_profile(args.push)
        push_host = f"{push_profile['User']}@{push_profile['Host']}"
        push_port = push_profile["Port"]
        header("Push to remote server", char="=")
        log("INFO", f"push profile         : {args.push}")
        log(
            "INFO",
            f"push target          : {push_host}"
            f"{f' (port {push_port})' if push_port else ''}",
        )
        log("INFO", f"push-root key        : {args.push_root}")
        log("INFO", f"remote path          : {push_remote}")
        log(
            "INFO",
            f"password .env key    : {args.push_server_password} "
            f"({'<set>' if push_password else '<empty -> key-based auth>'})",
        )
        log("INFO", f"parallel jobs        : {args.jobs}")
        log("INFO", f"models to push       : {len(succeeded)}")
        if args.dry_run:
            log("INFO", "dry-run -- no files will be sent")
        print()

        pushed, skipped, push_failed = push_models_to_server(
            repos=succeeded,
            local_root=root,
            profile=push_profile,
            password=push_password,
            remote_root=push_remote,
            jobs=args.jobs,
            dry_run=args.dry_run,
        )

        print()
        header("Push Summary", char="-")
        log("INFO", f"files pushed  : {pushed}")
        log("INFO", f"files skipped : {skipped} (already on remote)")
        if push_failed:
            log("FAIL", f"files failed  : {push_failed}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
