#!/usr/bin/env python3
"""Cascade LCP training artifacts: SendServer -> LOCAL -> ReceiveServer.

Arguments (every flag must be explicit on the CLI -- no hidden defaults):
    --sender PROFILE          REQUIRED. .env profile for the DOWNLOAD source.
    --receiver PROFILE        REQUIRED. .env profile for the UPLOAD destination.
    -m MODULE                 Module to process (builder/predictor/all).
    -e EXPERIMENT             Experiment dir name, or 'all' to auto-discover.
    -d DATASET                Dataset for Loss_prepare pass (repeatable; 'all' default).
    -p PASSWORD               Receive SSH password override (profile PWD if omitted).
    -i PATTERN                Artifact pattern to skip (repeatable).
    -j N                      Parallel worker count ('auto' default).
    --send-ssh SSH_STR        Override sender SSH string from .env.
    --send-storage-root PATH  Override sender StorageRoot from .env.
    --receive-ssh SSH_STR     Override receiver SSH string from .env.
    --receive-storage-root PATH  Override receiver StorageRoot from .env.
    --local-base PATH         Local staging dir (default: ./EXPERIMENT/lcp).
    --keep-checkpoints        Do not delete local checkpoints after upload.
    --dry-run                 Print commands; touch no files.

Effects:
    Sender   : resolved from ./.env[<sender>{User,Host,Port,StorageRoot}]
    Receiver : resolved from ./.env[<receiver>{User,Host,Port,PWD,StorageRoot}]
    Pipeline : PRECHECK -> DOWNLOAD -> UPLOAD -> CLEANUP (per experiment)
    Parallel : experiments cascade concurrently (per -j)
    Idempotent : skip download+upload when all checkpoints already on Receive

Usage::

    python3 LabToLocalToCompusServer.py --sender LabServer --receiver CampusServer --dry-run
    python3 LabToLocalToCompusServer.py --sender LabServer --receiver CampusServer -m builder -e DEMO
    python3 LabToLocalToCompusServer.py --sender LabServer --receiver CampusServer -j 4

----------------------------------------------------------------------
Detailed description
----------------------------------------------------------------------
Unlike ``LabToCampusServer.py`` (Send -> Receive in a single hop, run
ON Send), this script orchestrates a THREE-NODE cascade from your
LOCAL workstation: it pulls artifacts down from the SendServer to the
local staging dir, then pushes them up to the ReceiveServer.

+----------------+-------------------------------------------------+
| Role           | Sender profile (DOWNLOAD source)                |
| Selected by    | ``--sender <Profile>`` (REQUIRED)               |
| Endpoint       | $<Sender>User @ $<Sender>{Host|IP} [: $<Sender>Port] |
| Storage root   | $<Sender>StorageRoot                            |
| Full src path  | <SenderStorageRoot>/<method>/<module>/<experiment>/|
+----------------+-------------------------------------------------+
| Role           | Receiver profile (UPLOAD destination)           |
| Selected by    | ``--receiver <Profile>`` (REQUIRED)             |
| Endpoint       | $<Receiver>User @ $<Receiver>{Host|IP}          |
|                | [: $<Receiver>Port]                             |
| Storage root   | $<Receiver>StorageRoot                          |
| Full tgt path  | <ReceiverStorageRoot>/<method>/<module>/<experiment>/|
+----------------+-------------------------------------------------+

Both servers SHARE the same relative path
``<method>/<module>/<experiment>/`` (encoded in every config's
``log.save_folder``). Only the storage ROOT differs between the two
servers -- this is what makes the cascaded checkpoints immediately
usable for follow-up training on the ReceiveServer with the SAME
config files.

IMPORTANT: ``EXPERIMENT/`` is part of the user-supplied storage root
(not auto-appended). The script joins ONLY ``/<method>`` after the
root, so you control the entire prefix. There is NO hardcoded
fallback for any path -- if a key is unset in ``./.env`` AND the
matching CLI flag is not passed, the script fails fast at startup.

Local staging mirrors the same relative layout, with the local CWD
playing the role of the storage root::

    ./EXPERIMENT/<method>/<module>/<experiment>/

----------------------------------------------------------------------
PRE-FLIGHT SETUP  (do this ONCE per workstation; cascade is .env-driven)
----------------------------------------------------------------------
This script ships TRAINING ARTIFACTS produced by configs that follow
the ``ModelLearn`` template. Two separate inputs must be in place
before you can run a cascade:

  (1) ``./.env`` at the repo root -- declares NAMED SERVER PROFILES
      (each one a flat ``<Profile><Attr>`` namespace; see Step 1) and
      optional default role bindings. This script picks the SOURCE
      profile via ``--sender <Profile>`` and the DESTINATION profile
      via ``--receiver <Profile>``; when omitted, ``DefaultSender`` /
      ``DefaultReceiver`` from ``./.env`` are used. There is no
      hardcoded fallback; missing values fail fast at startup.

  (2) Per-experiment YAML configs under ``configs/<method>/<dataset>/
      [<sub>/]train_<module>_*.yml`` -- they ARE the discovery source
      of truth. Their ``log.save_folder`` field encodes the
      experiment-name leaf that all three nodes share underneath
      their respective storage roots, which is what makes the
      cascaded checkpoints immediately re-trainable on Receive with
      the SAME config files.

Step 1 -- Configure ``./.env`` with NAMED SERVER PROFILES. Each
profile is a flat ``<Profile><Attr>`` namespace; only the attributes
relevant to a given role are consumed. For this script the SENDER
profile contributes ``User`` / ``Host`` (or ``IP``) / ``Port`` /
``StorageRoot``, and the RECEIVER profile contributes the same set
plus ``PWD``. Example minimal layout (see ``./.env`` for the live
template)::

    # Sender profile (DOWNLOAD source)
    LabServerIP             = 10.123.4.30
    LabServerUser           = sjia
    LabServerHost           =                             # OPTIONAL; falls
                                                          # back to <P>IP
    LabServerPort           =                             # OPTIONAL; empty ->
                                                          # ssh client default
    LabServerStorageRoot    = /Data/ReasoningNLCP/EXPERIMENT
    # NOTE: the Sender PWD is NOT consumed for the download half;
    # use SSH-key trust from your local workstation to the Sender.

    # Receiver profile (UPLOAD destination)
    CampusServerIP          = 10.120.48.27
    CampusServerUser        = alice
    CampusServerHost        = hpcfront.example.edu        # OPTIONAL; falls
                                                          # back to <P>IP
    CampusServerPort        =                             # OPTIONAL; empty ->
                                                          # ssh client default
    CampusServerPWD         = <password>                  # OPTIONAL; when set,
                                                          # sshpass is invoked
                                                          # for the UPLOAD half.
    CampusServerStorageRoot = /data/user/alice/ReasoningNLCP/EXPERIMENT

    # Optional ergonomics: argument-free invocation.
    DefaultSender   = LabServer
    DefaultReceiver = CampusServer

At startup the script resolves the chosen profiles into the five
transfer parameters that show up in the banner::

    sender                    -> ``--sender`` or ``$DefaultSender``
    receiver                  -> ``--receiver`` or ``$DefaultReceiver``
    send_ssh                  -> from ``$<Sender>{User,Host|IP,Port}``
    send_storage_root         -> ``$<Sender>StorageRoot``
    receive_ssh / password    -> from ``$<Receiver>{User,Host|IP,Port,PWD}``
    receive_storage_root      -> ``$<Receiver>StorageRoot``

Any CLI flag (``--sender`` / ``--receiver`` / ``--send-ssh`` /
``--send-storage-root`` / ``--receive-ssh`` / ``--receive-storage-root``
/ ``-p``) takes precedence; the ``.env`` values only seed the defaults
so the normal invocation can be argument-free::

    python3 LabToLocalToCompusServer.py        # uses .env profiles transparently
    python3 LabToLocalToCompusServer.py --sender LabServer --receiver CampusServer

Step 2 -- Author / curate per-experiment YAML configs.
Every training run is described by a YAML file under
``configs/<method>/<dataset>/[<sub>/]train_<module>_*.yml``. These
files MUST follow the ``ModelLearn`` template, available in the
companion ``lmbase`` package at
``third-part/lmbase/template/`` (see also the in-repo reference
``configs/ModelLearn/{main,model,training}.yml``). The 6 mandatory
top-level blocks are::

    data         dataset identity / loading
    environment  runtime context (seed, device, dotenv, num_workers)
    model        architecture definition (dimensions, layers, heads)
    training     optimization (batch_size, epochs, lr, ...)
    log          save_folder, log_path, checkpoint_path, intervals
    evaluation   eval datasets, metrics, eval_step_interval

The cascade keys off ONE field only::

    log:
      save_folder: ./EXPERIMENT/<method>/<module>/<experiment-name>

The ``Path(save_folder).name`` becomes the experiment-name leaf that
the trainer materialises on the Sender at
``<SenderStorageRoot>/<method>/<module>/<experiment-name>/``, that the
LOCAL stage materialises at
``./EXPERIMENT/<method>/<module>/<experiment-name>/``, and that this
script mirrors verbatim onto the Receiver at
``<ReceiverStorageRoot>/<method>/<module>/<experiment-name>/``.

Step 3 -- Run the cascade::

    python3 LabToLocalToCompusServer.py        # full sweep, .env defaults

----------------------------------------------------------------------
Pipeline
----------------------------------------------------------------------
Pipeline (per discovered experiment, processed in parallel via a worker
pool -- see ``-j/--jobs`` below):

    0. PRECHECK on the ReceiveServer: ``ssh ... find <exp_dir> -type f``
       inventories every file already on Receive under the experiment
       dir. The inventory drives two distinct optimizations:
         * if EVERY entry in ``CHECKPOINT_GLOBS`` already has a match
           on Receive, print [SKIP] and move straight to the next
           experiment -- no download, no upload, no overwrite;
         * otherwise the inventory is threaded into the UPLOAD step
           so the upload skips files already on Receive and ships only
           the missing ones (per-file scp, NO ``rm -rf`` wipe).
       Read-only probe; runs even in --dry-run mode.
    1. SCP-DOWN from the SendServer
       (endpoint + root from ``./.env``)
       the experiment's ``checkpoints/checkpoint_best-*.pt`` and
       ``checkpoints/checkpoint_best_eval-*.pt`` (epoch/step suffixes
       resolved via remote glob expansion) plus the ``logs/`` dir
       into the local ``./EXPERIMENT/<method>/<module>/<experiment>/``.
       Already-present files are SKIP-ed (idempotent).
    2. UPLOAD to the ReceiveServer
       (endpoint + root from ``./.env``; sshpass + -p when password
       is set). Per-file delta sync against the PRECHECK inventory:
       files already on Receive are SKIPPED (no overwrite); only
       the genuinely missing files are scp'd, each into the same
       relative layout under the experiment dir. Required parent
       dirs are created with a single batched ``ssh ... mkdir -p``
       call first.
    3. DELETE the local ``./EXPERIMENT/<method>/<module>/<experiment>/checkpoints/``
       directory to free disk space before moving on to the next item.
       Logs and Loss_prepare.json files are KEPT locally; only the
       heavyweight checkpoints are pruned between iterations.
    4. Continue with the next experiment.

Parallelism (``-j/--jobs``):
    Steps 0-3 above are run end-to-end by ONE worker for ONE experiment;
    multiple workers run concurrently, one experiment each. Default
    ``-j auto`` chooses ``min(4, N_experiments)``. Pass ``-j 1`` for
    strict sequential mode (live-streaming scp output). In parallel
    mode each worker buffers its log block and the main thread flushes
    whole blocks atomically as workers finish, so log lines from
    different experiments do NOT interleave. WARNING: parallel mode
    keeps up to ``N`` experiments' checkpoints on local disk
    simultaneously -- size your ``--local-base`` accordingly.

In addition to per-experiment artifacts, the script also fetches +
uploads each dataset's ``<dataset>_Loss_prepare.json`` (small, kept
locally -- no cleanup). The same two-tier precheck applies:
  * if the JSON is already on the ReceiveServer, the cascade is
    skipped entirely;
  * otherwise, if it already exists locally, only the download is
    skipped and the upload still runs.

Discovery convention (rooted at ``configs/<method>/`` -- default method
``lcp``)::

    configs/<method>/<dataset>/[<sub>/]train_<module>_*.yml
        -> experiment name == basename of log.save_folder in that YAML
        -> module          == "builder" or "predictor"
        -> dataset         == FIRST path component under configs/<method>/

----------------------------------------------------------------------
Usage (run on your LOCAL workstation)
----------------------------------------------------------------------
With ``./.env`` configured, NO arguments are required at all.

    # Recommended: cascade every builder + predictor experiment using
    # the .env defaults (Send / Receive endpoints, both storage roots).
    python3 LabToLocalToCompusServer.py -p 'YourCampusPassword'

    # Just the builder module.
    python3 LabToLocalToCompusServer.py -m builder -p 'YourCampusPassword'

    # A single named experiment (skip the auto-discovery sweep).
    python3 LabToLocalToCompusServer.py -m builder -e GSM8K_Qwen2.5-0.5B_6level -p '...'

    # Override the SendServer storage root from ``./.env`` (must match
    # the ``-s`` the SendServer training was launched with).
    python3 LabToLocalToCompusServer.py -p '...' --send-storage-root /Data2/ReasoningNLCP/EXPERIMENT

    # Override the ReceiveServer storage root from ``./.env``.
    python3 LabToLocalToCompusServer.py -p '...' --receive-storage-root /data/user/other/ReasoningNLCP/EXPERIMENT

    # Pick a different sender / receiver profile from ``./.env`` at runtime.
    python3 LabToLocalToCompusServer.py --sender LabServer --receiver CampusServer

    # Keep the local checkpoints (skip the post-upload cleanup).
    python3 LabToLocalToCompusServer.py -p '...' --keep-checkpoints

    # Dry-run: print every scp / ssh / rm command that WOULD be issued
    # but execute nothing.
    python3 LabToLocalToCompusServer.py -p '...' --dry-run

    # Force strict sequential mode (live-streaming scp output, one
    # experiment at a time, smallest local disk footprint).
    python3 LabToLocalToCompusServer.py -p '...' -j 1

    # Override the parallel worker count (default 'auto' = min(4, N)).
    python3 LabToLocalToCompusServer.py -p '...' -j 8

Arguments:
    --sender                Named server profile (from ``./.env``)
                            describing the DOWNLOAD source. Resolves
                            SSH endpoint + storage root from
                            ``<P>{IP,User,Host,Port,StorageRoot}``.
                            Default: ``./.env`` key ``DefaultSender``.
    --receiver              Named server profile (from ``./.env``)
                            describing the UPLOAD destination.
                            Resolves SSH endpoint + password +
                            storage root from
                            ``<P>{IP,User,Host,Port,PWD,StorageRoot}``.
                            Default: ``./.env`` key ``DefaultReceiver``.
    -p / --password         ReceiveServer SSH login password used for the
                            UPLOAD half of the cascade. Default:
                            ``--receiver`` profile key ``<P>PWD``.
                            Required when the receive host needs
                            password auth (the common case). When
                            omitted (env unset and CLI not passed),
                            plain ssh/scp is invoked for the upload
                            and any auth failure surfaces at runtime
                            (no preemptive validation). Requires
                            ``sshpass`` on the local machine
                            (macOS: ``brew install hudochenkov/sshpass/sshpass``;
                            Debian/Ubuntu: ``sudo apt-get install sshpass``).
    -m / --module           "builder", "predictor", or "all" (default
                            "all": runs builder first, then predictor).
    -e / --experiment       Specific experiment directory name, or "all"
                            (default "all": auto-discover from configs).
    -d / --dataset          Specific dataset for the Loss_prepare.json
                            pass; may be given multiple times. Pass
                            "all" (default) to auto-discover datasets
                            from configs/<method>/<dataset>/train_<module>_*.yml.
                            Pass "" (empty string) to skip the
                            Loss_prepare pass entirely.
    --send-ssh              SendServer SSH connection string in the form
                            'ssh [-p PORT] user@host'. Default: built
                            from ``--sender`` profile keys
                            ``<P>{User,Host|IP,Port}``. Inline override
                            for the rare ad-hoc case.
    --send-storage-root     Storage root on the SendServer. The SendServer
                            base becomes ``<root>/<method>``. Default:
                            ``--sender`` profile key ``<P>StorageRoot``.
                            MUST include the ``EXPERIMENT/`` segment if
                            your save_folder does NOT (the script appends
                            only ``/<method>``).
    --receive-ssh           ReceiveServer SSH connection string. Default:
                            built from ``--receiver`` profile keys
                            ``<P>{User,Host|IP,Port}``. Inline override
                            for the rare ad-hoc case.
    --receive-storage-root  Storage root on the ReceiveServer. The receive
                            base becomes ``<root>/<method>`` (same
                            relative layout as the SendServer, so the
                            SAME config files work post-cascade).
                            Default: ``--receiver`` profile key
                            ``<P>StorageRoot``.
    --local-base            Override the LOCAL staging base dir.
                            Default: ``./EXPERIMENT/<method>``.
    --keep-checkpoints      Do NOT delete local checkpoints/ after a
                            successful upload (overrides the default
                            one-at-a-time disk discipline).
    --dry-run               Print every external command the script
                            WOULD invoke (scp / ssh / rm / mkdir) but
                            do not execute any of them.

Failure policy:
    * If the SendServer download for an experiment fails, the script logs
      [PARTIAL], leaves any partial local files in place, and CONTINUES
      with the next experiment.
    * If the ReceiveServer upload fails, the local checkpoints are
      PRESERVED (cleanup is skipped) so the user can retry. The script
      still continues with the next experiment so a transient hiccup on
      one doesn't block the rest.
    * Cleanup ONLY runs after a fully successful upload of the current
      experiment.
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import os
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from dotenv import load_dotenv

# --- .env loading ------------------------------------------------------
# Load credentials / profile attributes from ``./.env`` at the repo
# root BEFORE any default is computed below. We never hard-code
# username, hostname or password in this source file -- everything
# sensitive flows through ``./.env``. ``override=False`` so a real env
# var (exported in the shell) wins over the file, matching typical CI
# conventions.
load_dotenv(
    dotenv_path=Path(__file__).resolve().parent / ".env",
    override=False,
)

# Default role bindings. When ``--sender`` / ``--receiver`` are omitted
# at the CLI, these supply the profile names; leave the env keys empty
# to force the operator to pass the flags explicitly.
DEFAULT_SENDER = ""
DEFAULT_RECEIVER = ""

# Attributes the server-profile resolver knows about. Keys are read as
# ``<Profile><Attr>`` from the environment. Same shape across all four
# .env-driven scripts so a single profile in ``./.env`` works everywhere.
_PROFILE_ATTRS = (
    "IP",
    "User",
    "Host",
    "Port",
    "PWD",
    "StorageRoot",
    "CodeRoot",
    "ModelPath",
)


def load_server_profile(name: str) -> dict[str, str]:
    """Resolve a named server profile from ``./.env`` into a flat dict.

    Reads every ``<name><Attr>`` env key and returns ``{Attr: value}``
    with empty strings for unset attributes. ``Host`` falls back to
    ``IP`` so callers can use a single ssh-target field uniformly.

    Raises:
        SystemExit: when ``name`` is empty or no profile attribute is
            populated under that prefix (typo / missing .env block).
    """
    if not name:
        raise SystemExit(
            "[FATAL] empty server profile name; pass --sender/--receiver "
            "<Profile>. There is no default -- the flag is REQUIRED."
        )
    out = {a: (os.environ.get(f"{name}{a}") or "").strip() for a in _PROFILE_ATTRS}
    if not any(out.values()):
        raise SystemExit(
            f"[FATAL] server profile {name!r} not found in ./.env -- "
            f"expected at least one of: "
            + ", ".join(f"{name}{a}" for a in _PROFILE_ATTRS)
        )
    if not out["Host"]:
        out["Host"] = out["IP"]
    return out


def profile_ssh(profile: dict[str, str]) -> str:
    """Build ``ssh [-p PORT] USER@HOST`` from a resolved profile dict.

    Returns ``""`` when User or Host is missing so the caller can
    fail-fast with a single, actionable message naming both the missing
    env key and the matching CLI override.
    """
    user, host, port = profile["User"], profile["Host"], profile["Port"]
    if not user or not host:
        return ""
    return f"ssh -p {port} {user}@{host}" if port else f"ssh {user}@{host}"


DEFAULT_LOCAL_BASE = "./EXPERIMENT/lcp"
DEFAULT_CONFIGS_ROOT = Path("./configs/lcp")
# Shared relative subpath under BOTH servers' storage roots.
# It encodes the project-wide convention used by every YAML's
# ``log.save_folder`` -- appended identically to both send and receive
# roots so the same configs round-trip without surgery. The storage
# roots themselves (``$<Sender>StorageRoot`` / ``$<Receiver>StorageRoot``
# in ``./.env``) ALREADY include the ``EXPERIMENT/`` segment, so only
# the per-method subdir is appended here.
EXPERIMENT_SUBPATH = "lcp"

# Checkpoint artifact patterns (NOT literal filenames). The trainer
# names checkpoints with an ``-epoch<E>-step<S>.pt`` suffix, e.g.
# ``checkpoint_best-epoch9-step17500.pt`` /
# ``checkpoint_best_eval-epoch9-step18500.pt``. These two globs match
# the two best-checkpoint families independently — the trailing ``-``
# after ``best`` is what keeps the first pattern from accidentally
# matching ``checkpoint_best_eval-*`` (since ``-`` != ``_``).
#
# Remote-side glob expansion: scp launches a shell on the remote host
# (via ssh) to resolve the source path, so ``*`` is expanded REMOTELY.
# Locally we single-quote the source argument when building the scp
# command, otherwise the local zsh would try to expand ``*`` against
# the local filesystem (and fail).
CHECKPOINT_GLOBS = [
    "checkpoints/checkpoint_best_eval-*.pt",
    "checkpoints/checkpoint_best-*.pt",
]
LOGS_DIR = "logs"
CHECKPOINTS_DIR = "checkpoints"
LOSS_PREPARE_SUFFIX = "_Loss_prepare.json"

VALID_MODULES = ("builder", "predictor")
ALL_KEYWORD = "all"

# ----------------------------------------------------------------------
# Display helpers -- consistent, scan-friendly tagged logging.
# Every line emitted by the cascade uses a fixed-width ``[TAG ]`` prefix
# so the operator can grep / eyeball the run in seconds:
#   [ DL ]   downloading from SendServer
#   [ UP ]   uploading to ReceiveServer
#   [SKIP]   artifact already present / nothing to do
#   [ OK ]   step succeeded
#   [DONE]   end-to-end cascade succeeded for one item
#   [MISS]   remote artifact not found (continue)
#   [WARN]   non-fatal anomaly (continue)
#   [FAIL]   step failed (stop the current item, continue with next)
#   [CLEAN]  local cleanup action
#   [KEEP]   local cleanup skipped intentionally
#   [INFO]   neutral status / configuration line
# ----------------------------------------------------------------------
# Standard rule width for headers / separators.
LINE_W = 78
# Fixed width of the [TAG ] column.
TAG_W = 6

# Module-level lock used in parallel mode to atomically flush each
# worker's captured output buffer to the real terminal. Without it,
# two workers' multi-line blocks could interleave on stdout.
_OUTPUT_LOCK = threading.Lock()


def _tag(tag: str) -> str:
    """Return a fixed-width ``[TAG ]`` prefix string."""
    return f"[{tag.center(TAG_W - 2)}]"


def log(tag: str, msg: str, *, indent: int = 0, out=None) -> None:
    """Print one tagged log line.

    ``indent`` is a leading-space count. ``out`` selects the destination
    stream — defaults to ``sys.stdout`` (sequential mode); callers
    running inside a parallel worker pass a per-worker ``io.StringIO``
    so the line is buffered until the worker finishes.
    """
    target = out if out is not None else sys.stdout
    print(f"{' ' * indent}{_tag(tag)} {msg}", file=target)


def hr(char: str = "=", *, out=None) -> None:
    """Print a horizontal rule at ``LINE_W`` width."""
    target = out if out is not None else sys.stdout
    print(char * LINE_W, file=target)


def header(title: str, char: str = "=", *, out=None) -> None:
    """Print ``title`` between two horizontal rules."""
    target = out if out is not None else sys.stdout
    hr(char, out=target)
    print(title, file=target)
    hr(char, out=target)


def sub_header(title: str, *, out=None) -> None:
    """Print a thin section divider (used inside an experiment)."""
    target = out if out is not None else sys.stdout
    print(file=target)
    print("-" * LINE_W, file=target)
    print(f"  {title}", file=target)
    print("-" * LINE_W, file=target)


# ======================================================================
# CLI parsing
# ======================================================================
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for three-node cascade transfer."""
    parser = argparse.ArgumentParser(
        description=(
            "Cascade LCP artifacts: SendServer -> LOCAL -> ReceiveServer, "
            "one experiment at a time, with post-upload checkpoint cleanup. "
            "Both endpoints are configured via NAMED SERVER PROFILES in "
            "``./.env`` (selected at the CLI with ``--sender <Profile>`` / "
            "``--receiver <Profile>``) -- this works for ANY two hosts, "
            "not just Lab/Campus."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sender",
        default="",
        metavar="PROFILE",
        help=(
            "Named server profile (from ``./.env``) describing the "
            "DOWNLOAD source. Resolves SSH endpoint + storage root "
            "from ``<P>{IP,User,Host,Port,StorageRoot}``. REQUIRED."
        ),
    )
    parser.add_argument(
        "--receiver",
        default="",
        metavar="PROFILE",
        help=(
            "Named server profile (from ``./.env``) describing the "
            "UPLOAD destination. Resolves SSH endpoint + password + "
            "storage root from ``<P>{IP,User,Host,Port,PWD,StorageRoot}``. "
            "REQUIRED."
        ),
    )
    parser.add_argument(
        "-p",
        "--password",
        default=None,
        help=(
            "ReceiveServer SSH login password (used for the UPLOAD half via "
            "sshpass). Default: ``--receiver`` profile key ``<P>PWD``. "
            "Pass ``-p ''`` to force key-based auth and ignore the profile "
            "value."
        ),
    )
    parser.add_argument(
        "-m",
        "--module",
        default=ALL_KEYWORD,
        choices=[*VALID_MODULES, ALL_KEYWORD],
        help=f"Module to process. Default: '{ALL_KEYWORD}'.",
    )
    parser.add_argument(
        "-e",
        "--experiment",
        default=ALL_KEYWORD,
        help=(
            "Experiment dir name under EXPERIMENT/lcp/<module>/, or "
            "'all' to auto-discover from configs/lcp/**/train_<module>_*.yml."
        ),
    )
    parser.add_argument(
        "-d",
        "--dataset",
        action="append",
        default=None,
        metavar="DATASET",
        help=(
            "Dataset for the Loss_prepare.json pass. May be given "
            "multiple times. Pass 'all' (default when omitted) to "
            "auto-discover. Pass '' (empty) to skip the pass entirely."
        ),
    )
    parser.add_argument(
        "--send-ssh",
        default="",
        help=(
            "SendServer SSH string of the form 'ssh [-p PORT] USER@HOST'. "
            "Default: built from ``--sender`` profile keys "
            "``<P>{User,Host|IP,Port}``. Inline override only."
        ),
    )
    parser.add_argument(
        "--send-storage-root",
        default="",
        help=(
            "SendServer storage root. The SendServer base becomes "
            f"``<root>/{EXPERIMENT_SUBPATH}``. Default: ``--sender`` "
            "profile key ``<P>StorageRoot``. Inline override only."
        ),
    )
    parser.add_argument(
        "--receive-ssh",
        default="",
        help=(
            "ReceiveServer SSH string of the form 'ssh [-p PORT] USER@HOST'. "
            "Default: built from ``--receiver`` profile keys "
            "``<P>{User,Host|IP,Port}``. Inline override only."
        ),
    )
    parser.add_argument(
        "--receive-storage-root",
        default="",
        help=(
            "ReceiveServer storage root. The receive base becomes "
            f"``<root>/{EXPERIMENT_SUBPATH}`` -- SAME relative layout "
            "as the SendServer so configs round-trip unchanged. Default: "
            "``--receiver`` profile key ``<P>StorageRoot``. Inline "
            "override only."
        ),
    )
    parser.add_argument(
        "--local-base",
        default=DEFAULT_LOCAL_BASE,
        help=f"Local staging base dir. Default: '{DEFAULT_LOCAL_BASE}'.",
    )
    parser.add_argument(
        "-i",
        "--ignore",
        action="append",
        default=[],
        metavar="PATTERN",
        help=(
            "Artifact pattern to SKIP, relative to the experiment dir "
            "(same matching rules as run_scp.py: exact / parent / "
            "fnmatch glob). Repeatable. Example: '-i checkpoints/*best.pt'."
        ),
    )
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Do not delete local checkpoints/ after a successful upload.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        default="auto",
        metavar="N",
        help=(
            "Parallel worker count for the experiment cascade. Each "
            "worker handles one experiment end-to-end (PRECHECK -> "
            "DOWNLOAD -> UPLOAD -> CLEANUP). 'auto' (default) picks "
            "min(4, len(experiments)). Pass an int to override (e.g. "
            "'-j 1' = strict sequential, '-j 8' = up to 8 concurrent "
            "experiments). NOTE: parallel mode keeps up to N "
            "experiments' checkpoints on local disk simultaneously — "
            "size your --local-base accordingly."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print every external command without executing it.",
    )
    return parser.parse_args()


# ======================================================================
# Tiny helpers
# ======================================================================
# ``port == ""`` semantics: every helper returns the empty string (NOT
# ``"22"``) when no explicit port is configured. Call sites then emit
# a bare ``ssh user@host`` and let the ssh client honour
# ``~/.ssh/config`` -- silently injecting ``22`` would mask a non-22
# port the operator declared in their config.
def parse_ssh(ssh_str: str) -> tuple[str, str]:
    """Extract ``(port, user@host)`` from ``ssh [-p PORT] user@host``.

    Returns ``port == ""`` when the SSH string has no ``-p`` flag, so
    downstream call sites can omit ``-p`` / ``-P`` entirely and let the
    ssh client honour ``~/.ssh/config``.

    Raises:
        SystemExit: when ``ssh_str`` does not contain a ``user@host``
            token (i.e. the operator passed an obviously malformed
            ``--*-ssh`` flag or env value).
    """
    port_match = re.search(r"-p\s+(\d+)", ssh_str)
    port = port_match.group(1) if port_match else ""
    host_match = re.search(r"(\S+@\S+)", ssh_str)
    if not host_match:
        raise SystemExit(f"[ERROR] Cannot parse 'user@host' from: {ssh_str!r}")
    return port, host_match.group(1)


def ssh_port_flag(port: str) -> str:
    """Return ``" -p N"`` (with leading space) or ``""`` when port empty."""
    return f" -p {port}" if port else ""


def scp_port_flag(port: str) -> str:
    """SCP equivalent of :func:`ssh_port_flag` (uses ``-P``)."""
    return f" -P {port}" if port else ""


def port_suffix(port: str) -> str:
    """Return ``"  (port N)"`` for banner display when ``port`` is set."""
    return f"  (port {port})" if port else ""


def build_auth_prefix(password: str | None) -> str:
    """Return the ``sshpass -p '...'`` prefix, or ``""`` when no password.

    No preemptive ``sshpass`` install check -- if the user supplies a
    password but ``sshpass`` is missing, the shell will report it at
    runtime.
    """
    if not password:
        return ""
    escaped = password.replace("'", "'\\''")
    return f"sshpass -p '{escaped}' "


def remote_glob_has_match(
    *,
    glob_path: str,
    auth: str,
    user_host: str,
    port: str,
) -> bool:
    """Probe a remote host via ssh; True iff ``ls -1 <glob_path>`` lists
    at least one entry.

    ``glob_path`` may contain shell globs — they are expanded REMOTELY
    by the ssh-spawned shell (not by the local zsh). The probe is
    purely read-only and cheap, so it is issued unconditionally — even
    in --dry-run mode — because its sole purpose is to AVOID wasteful
    download/upload cascades when the artifact is already on Receive.
    """
    cmd = (
        f"{auth}ssh{ssh_port_flag(port)} {user_host} "
        f'"ls -1 {glob_path} 2>/dev/null | head -n 1"'
    )
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
    return bool(proc.stdout.strip())


def remote_list_files(
    *,
    remote_dir: str,
    auth: str,
    user_host: str,
    port: str,
) -> set[str]:
    """Recursive file inventory of ``remote_dir`` on the receive host.

    Returns a set of POSIX paths RELATIVE to ``remote_dir`` (no
    leading ``./``). Empty set if the directory does not exist on
    the remote (silently). Read-only; runs unconditionally (incl.
    --dry-run) because the result drives delta-upload decisions and
    avoiding wasteful re-uploads is the whole point of this probe.

    Implementation: ``cd <dir> && find . -type f`` over a single ssh
    invocation. Falls back to an empty result when the dir is missing.
    """
    cmd = (
        f"{auth}ssh{ssh_port_flag(port)} {user_host} "
        f"\"if [ -d '{remote_dir}' ]; then cd '{remote_dir}' && find . -type f; fi\""
    )
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
    out: set[str] = set()
    for line in proc.stdout.splitlines():
        rel = line.strip()
        if not rel:
            continue
        if rel.startswith("./"):
            rel = rel[2:]
        out.add(rel)
    return out


def run(cmd: str, *, dry_run: bool = False, out=None) -> int:
    """Run a shell command and return its exit code.

    ``out`` selects where the command echo + child output go:
      * ``None`` / ``sys.stdout`` → stream child output to the terminal
        in real time (sequential mode).
      * a ``StringIO`` → capture child stdout+stderr and append to the
        buffer (parallel mode); avoids interleaved output between
        concurrent workers.

    In dry-run mode, prints the command (prefixed with [DRY]) and
    returns 0 without invoking the shell.
    """
    target = out if out is not None else sys.stdout
    prefix = "[DRY] $ " if dry_run else "$ "
    print(f"{prefix}{cmd}", file=target)
    if dry_run:
        return 0
    if target is sys.stdout:
        return subprocess.call(cmd, shell=True)
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
    if proc.stdout:
        target.write(proc.stdout)
        if not proc.stdout.endswith("\n"):
            target.write("\n")
    if proc.stderr:
        target.write(proc.stderr)
        if not proc.stderr.endswith("\n"):
            target.write("\n")
    return proc.returncode


def _matches_ignore(rel_path: str, patterns: list[str]) -> str | None:
    """Return the first -i pattern that matches ``rel_path``, else None.

    Matching rules (same as run_scp.py):
      * exact equality, OR
      * the pattern is a parent directory of ``rel_path``, OR
      * ``fnmatch.fnmatchcase`` glob match.
    """
    rp = rel_path.strip("/")
    for p in patterns:
        pat = p.strip("/")
        if not pat:
            continue
        if rp == pat or rp.startswith(pat + "/"):
            return p
        if fnmatch.fnmatchcase(rp, pat):
            return p
    return None


# ======================================================================
# Config discovery
# ======================================================================
def discover_experiments(module: str) -> list[str]:
    """configs/lcp/**/train_<module>_*.yml -> [experiment_name, ...].

    Experiment name is the basename of ``log.save_folder`` in each YAML --
    the same source of truth used by the trainer when creating the
    on-disk experiment directory.

    Raises:
        FileNotFoundError: when ``DEFAULT_CONFIGS_ROOT`` does not exist
            or is not a directory. Failing loudly here is intentional:
            a missing configs root makes the whole YAML-driven
            candidate set unknowable, and silently returning ``[]``
            would let downstream code mis-classify every dir as
            ``no-yaml-anchor``.
        KeyError: when a matched ``train_*.yml`` lacks the required
            ``log.save_folder`` field (used to derive the experiment
            name).
    """
    if not DEFAULT_CONFIGS_ROOT.is_dir():
        raise FileNotFoundError(
            f"Configs root not found or not a directory: {DEFAULT_CONFIGS_ROOT}. "
            f"Check that ``./configs/lcp`` exists in the current repo checkout."
        )
    prefix = f"train_{module}_"
    seen: set[str] = set()
    out: list[str] = []
    for yml in sorted(DEFAULT_CONFIGS_ROOT.rglob(f"{prefix}*.yml")):
        with yml.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        try:
            save_folder = cfg["log"]["save_folder"]
        except (KeyError, TypeError) as exc:
            raise KeyError(
                f"YAML {yml} is missing required field 'log.save_folder' "
                f"(needed to derive the experiment name); fix the config "
                f"or remove the stale file from {DEFAULT_CONFIGS_ROOT}."
            ) from exc
        name = Path(save_folder).name
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def discover_datasets(module: str) -> list[str]:
    """configs/lcp/<dataset>/[**/]train_<module>_*.yml -> [dataset, ...].

    Raises:
        FileNotFoundError: when ``DEFAULT_CONFIGS_ROOT`` does not exist
            or is not a directory (same rationale as
            :func:`discover_experiments`).
    """
    if not DEFAULT_CONFIGS_ROOT.is_dir():
        raise FileNotFoundError(
            f"Configs root not found or not a directory: {DEFAULT_CONFIGS_ROOT}. "
            f"Check that ``./configs/lcp`` exists in the current repo checkout."
        )
    prefix = f"train_{module}_"
    seen: set[str] = set()
    out: list[str] = []
    for yml in sorted(DEFAULT_CONFIGS_ROOT.rglob(f"{prefix}*.yml")):
        rel = yml.relative_to(DEFAULT_CONFIGS_ROOT)
        if len(rel.parts) < 2:
            continue
        ds = rel.parts[0]
        if ds in seen:
            continue
        seen.add(ds)
        out.append(ds)
    return out


# ======================================================================
# Per-experiment cascade: DOWNLOAD -> UPLOAD -> CLEANUP
# ======================================================================
def download_experiment(
    module: str,
    experiment: str,
    *,
    src_user_host: str,
    src_port: str,
    src_base: str,
    local_base: Path,
    ignore_patterns: list[str],
    dry_run: bool,
    out=None,
) -> tuple[bool, list[str]]:
    """SCP-down checkpoints + logs from the source remote.

    Returns ``(ok, missing)``. ``ok`` is True iff at least one artifact
    is locally present after the call (so the upload step has something
    to ship); ``missing`` lists relative paths that could not be fetched.
    """
    remote_dir = f"{src_base}/{module}/{experiment}"
    local_dir = local_base / module / experiment
    local_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    fetched_any = False

    # Checkpoints (glob-based: filenames include epoch/step suffixes).
    # For each glob pattern, we first probe LOCAL for any already-present
    # match (idempotent skip), then issue a single scp with the remote
    # glob — relying on the REMOTE shell (invoked by scp via ssh) to
    # expand ``*``. The local-shell argument is single-quoted so zsh
    # passes it through verbatim instead of trying its own glob match.
    for glob_pattern in CHECKPOINT_GLOBS:
        hit = _matches_ignore(glob_pattern, ignore_patterns)
        if hit is not None:
            log("SKIP", f"{glob_pattern}  (ignored by -i {hit!r})", indent=2, out=out)
            continue

        # ``rel_dir`` is the directory portion of the glob (e.g.
        # ``'checkpoints'``); ``basename_pat`` is the filename pattern
        # (e.g. ``'checkpoint_best-*.pt'``).
        rel_dir = Path(glob_pattern).parent
        basename_pat = Path(glob_pattern).name
        local_ckpt_dir = local_dir / rel_dir

        # Idempotent skip: any non-empty local file matching the pattern.
        existing = (
            [
                p
                for p in local_ckpt_dir.glob(basename_pat)
                if p.is_file() and p.stat().st_size > 0
            ]
            if local_ckpt_dir.is_dir()
            else []
        )
        if existing:
            for p in existing:
                log(
                    "SKIP",
                    f"{p.relative_to(local_dir)}  (already present)",
                    indent=2,
                    out=out,
                )
            fetched_any = True
            continue

        local_ckpt_dir.mkdir(parents=True, exist_ok=True)
        # Single-quote the source so the LOCAL shell does not glob-expand;
        # the REMOTE shell (spawned by scp via ssh) does the expansion.
        remote_src = f"'{src_user_host}:{remote_dir}/{glob_pattern}'"
        log("DL", f"{glob_pattern}", indent=2, out=out)
        rc = run(
            f"scp{scp_port_flag(src_port)} {remote_src} {local_ckpt_dir}/",
            dry_run=dry_run,
            out=out,
        )
        if rc != 0:
            log(
                "MISS",
                f"{glob_pattern}  (scp exit={rc}, remote not found?)",
                indent=2,
                out=out,
            )
            missing.append(glob_pattern)
            continue
        # In dry-run mode we cannot verify file presence; treat as success.
        if dry_run:
            fetched_any = True
            continue
        # Verify at least one file actually landed (remote glob may have
        # matched nothing, in which case scp can still return 0 silently
        # on some implementations).
        landed = [
            p
            for p in local_ckpt_dir.glob(basename_pat)
            if p.is_file() and p.stat().st_size > 0
        ]
        if landed:
            for p in landed:
                size_mb = p.stat().st_size / (1024 * 1024)
                log(
                    "OK",
                    f"{p.relative_to(local_dir)}  ({size_mb:.1f} MB)",
                    indent=2,
                    out=out,
                )
            fetched_any = True
        else:
            log(
                "WARN",
                f"scp returned 0 but no files matched {glob_pattern}",
                indent=2,
                out=out,
            )
            missing.append(glob_pattern)

    # Logs dir (recursive, atomic).
    hit = _matches_ignore(LOGS_DIR, ignore_patterns)
    local_logs = local_dir / LOGS_DIR
    if hit is not None:
        log("SKIP", f"{LOGS_DIR}/  (ignored by -i {hit!r})", indent=2, out=out)
    elif local_logs.is_dir() and any(local_logs.iterdir()):
        log(
            "SKIP",
            f"{LOGS_DIR}/  (already non-empty at {local_logs})",
            indent=2,
            out=out,
        )
        fetched_any = True
    else:
        # Avoid scp's nesting trap by removing an empty stub and
        # writing into the parent.
        if local_logs.exists():
            try:
                local_logs.rmdir()
            except OSError:
                pass
        remote_src = f"{src_user_host}:{remote_dir}/{LOGS_DIR}"
        log("DL", f"{LOGS_DIR}/  (recursive)", indent=2, out=out)
        rc = run(
            f"scp -r{scp_port_flag(src_port)} {remote_src} {local_dir}",
            dry_run=dry_run,
            out=out,
        )
        if rc != 0:
            log("MISS", f"{LOGS_DIR}/  (scp exit={rc})", indent=2, out=out)
            missing.append(f"{LOGS_DIR}/")
        else:
            log("OK", f"{LOGS_DIR}/  fetched", indent=2, out=out)
            fetched_any = True

    return fetched_any, missing


def upload_experiment(
    module: str,
    experiment: str,
    *,
    local_base: Path,
    existing_remote: set[str],
    recv_auth: str,
    recv_user_host: str,
    recv_port: str,
    recv_base: str,
    dry_run: bool,
    out=None,
) -> bool:
    """Per-file delta upload of the experiment dir to the receive host.

    ``existing_remote`` is the inventory (POSIX paths relative to the
    receive experiment dir) computed by ``remote_list_files`` during
    PRECHECK. Files already present on Receive are SKIPPED — no
    ``rm -rf`` wipe, no full re-transfer of multi-GB checkpoints that
    round-tripped in an earlier run. Only the genuinely missing files
    are scp'd, and they are placed under the same relative layout.

    Returns True iff every required upload succeeded (or there was
    nothing to upload).
    """
    local_dir = local_base / module / experiment
    if not local_dir.is_dir() or not any(local_dir.iterdir()):
        log("SKIP", f"nothing to upload at {local_dir}", indent=2, out=out)
        return True

    remote_target = f"{recv_base}/{module}/{experiment}"

    # 1. Decide what to upload (delta against the receive inventory).
    to_upload: list[tuple[Path, str]] = []
    for local_file in sorted(local_dir.rglob("*")):
        if not local_file.is_file():
            continue
        rel = local_file.relative_to(local_dir).as_posix()
        if rel in existing_remote:
            log("SKIP", f"{rel}  (already on Receive)", indent=2, out=out)
            continue
        to_upload.append((local_file, rel))

    if not to_upload:
        log("OK", "receive already in sync; nothing to upload", indent=2, out=out)
        return True

    # 2. Batch-create every parent dir we are about to write into
    #    (single ssh round-trip instead of one per file).
    parents = sorted(
        {
            (
                f"{remote_target}/{Path(rel).parent.as_posix()}"
                if Path(rel).parent.as_posix() not in (".", "")
                else remote_target
            )
            for _, rel in to_upload
        }
    )
    log("INFO", f"mkdir -p (batch, {len(parents)} dir(s))", indent=2, out=out)
    mk_args = " ".join(f"'{p}'" for p in parents)
    rc = run(
        f"{recv_auth}ssh{ssh_port_flag(recv_port)} {recv_user_host} "
        f'"mkdir -p {mk_args}"',
        dry_run=dry_run,
        out=out,
    )
    if rc != 0:
        log("FAIL", f"batch mkdir failed (rc={rc})", indent=2, out=out)
        return False

    # 3. Upload each missing file individually; preserves the relative
    #    layout under remote_target without nesting.
    log(
        "INFO",
        f"transferring {len(to_upload)} missing file(s) to Receive",
        indent=2,
        out=out,
    )
    failed: list[str] = []
    for local_file, rel in to_upload:
        size_mb = local_file.stat().st_size / (1024 * 1024)
        log("UP", f"{rel}  ({size_mb:.1f} MB)", indent=2, out=out)
        rc = run(
            f"{recv_auth}scp{scp_port_flag(recv_port)} {local_file} "
            f"{recv_user_host}:'{remote_target}/{rel}'",
            dry_run=dry_run,
            out=out,
        )
        if rc != 0:
            log("FAIL", f"scp {rel} (rc={rc})", indent=2, out=out)
            failed.append(rel)
            continue
        log("OK", f"{rel}", indent=2, out=out)

    if failed:
        log(
            "FAIL",
            f"{len(failed)} file(s) failed to upload: {failed}",
            indent=2,
            out=out,
        )
        return False
    log(
        "OK",
        f"uploaded {len(to_upload)} file(s) for {module}/{experiment}",
        indent=2,
        out=out,
    )
    return True


def cleanup_checkpoints(
    module: str,
    experiment: str,
    *,
    local_base: Path,
    dry_run: bool,
    out=None,
) -> None:
    """Delete local <local_base>/<module>/<experiment>/checkpoints/."""
    ckpt_dir = local_base / module / experiment / CHECKPOINTS_DIR
    if not ckpt_dir.exists():
        log("SKIP", f"no local {ckpt_dir} to clean", indent=2, out=out)
        return
    log("CLEAN", f"rm -rf {ckpt_dir}", indent=2, out=out)
    if dry_run:
        return
    shutil.rmtree(ckpt_dir, ignore_errors=True)


# ======================================================================
# Per-Loss_prepare cascade
# ======================================================================
def cascade_loss_prepare(
    module: str,
    dataset: str,
    *,
    src_user_host: str,
    src_port: str,
    src_base: str,
    local_base: Path,
    recv_auth: str,
    recv_user_host: str,
    recv_port: str,
    recv_base: str,
    dry_run: bool,
) -> bool:
    """Download + upload a single <dataset>_Loss_prepare.json (no cleanup).

    Returns True on a fully successful cascade (download + upload).
    Small JSON files — we don't delete them locally after upload.
    """
    fname = f"{dataset}{LOSS_PREPARE_SUFFIX}"
    remote_path = f"{src_base}/{module}/{fname}"
    local_path = local_base / module / fname

    sub_header(f"LOSS_PREPARE  module={module}  dataset={dataset}")
    log("INFO", f"send   : {src_user_host}:{remote_path}", indent=2)
    log("INFO", f"local : {local_path}", indent=2)
    recv_target = f"{recv_base}/{module}/{fname}"
    log("INFO", f"receive: {recv_user_host}:{recv_target}", indent=2)

    # 0. Pre-check Receive: if the JSON is already at the cascade target,
    #    there is nothing to download AND nothing to upload — short-circuit
    #    the whole cascade to avoid extra traffic / overwrites.
    if remote_glob_has_match(
        glob_path=recv_target,
        auth=recv_auth,
        user_host=recv_user_host,
        port=recv_port,
    ):
        log("SKIP", f"{fname}  (already on Receive at {recv_target})", indent=2)
        log("DONE", f"{fname} already cascaded (no-op)", indent=2)
        return True

    # 1. Download (skip if already present locally).
    if local_path.is_file() and local_path.stat().st_size > 0:
        log("SKIP", f"{fname}  (already at {local_path})", indent=2)
    else:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        remote_src = f"{src_user_host}:{remote_path}"
        log("DL", f"{fname}", indent=2)
        rc = run(
            f"scp{scp_port_flag(src_port)} {remote_src} {local_path}",
            dry_run=dry_run,
        )
        if rc != 0:
            log("MISS", f"{fname}  (download failed, rc={rc})", indent=2)
            if local_path.exists() and local_path.stat().st_size == 0:
                try:
                    local_path.unlink()
                except OSError:
                    pass
            return False
        log("OK", f"{fname}  downloaded", indent=2)

    # 2. Upload (mkdir -p, then scp).
    remote_parent = f"{recv_base}/{module}"
    log("INFO", f"mkdir -p {remote_parent}", indent=2)
    rc = run(
        f"{recv_auth}ssh{ssh_port_flag(recv_port)} {recv_user_host} "
        f"\"mkdir -p '{remote_parent}'\"",
        dry_run=dry_run,
    )
    if rc != 0:
        log("FAIL", f"mkdir -p {remote_parent} (rc={rc})", indent=2)
        return False
    log("UP", f"{fname}  ->  {recv_user_host}:{remote_parent}/", indent=2)
    rc = run(
        f"{recv_auth}scp{scp_port_flag(recv_port)} {local_path} "
        f"{recv_user_host}:{remote_parent}/",
        dry_run=dry_run,
    )
    if rc != 0:
        log("FAIL", f"{fname} upload (rc={rc})", indent=2)
        return False
    log("DONE", f"{fname} cascaded", indent=2)
    return True


# ======================================================================
# Per-experiment cascade (one worker handles ONE experiment)
# ======================================================================
def _resolve_jobs(jobs_arg: str, n: int) -> int:
    """Resolve the ``-j/--jobs`` argument to a concrete worker count.

    'auto' picks ``min(4, n)`` (capped at 4 to stay polite to SSH/network);
    any positive int passes through. Falls back to 1 when ``n == 0``.
    """
    if n <= 0:
        return 1
    if isinstance(jobs_arg, str) and jobs_arg.strip().lower() == "auto":
        return max(1, min(4, n))
    try:
        v = int(jobs_arg)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"--jobs must be 'auto' or a positive int, got {jobs_arg!r}"
        ) from exc
    if v < 1:
        raise SystemExit(f"--jobs must be >= 1, got {v}")
    return v


def _cascade_one_experiment(
    idx: int,
    total: int,
    module: str,
    experiment: str,
    *,
    args: argparse.Namespace,
    send_user_host: str,
    send_port: str,
    send_base: str,
    recv_auth: str,
    recv_user_host: str,
    recv_port: str,
    recv_base: str,
    local_base: Path,
    capture: bool,
) -> tuple[str, bool, bool, str]:
    """Run the full PRECHECK -> DOWNLOAD -> UPLOAD -> CLEANUP cascade for
    one experiment.

    Returns ``(label, partial, failed_upload, output_text)``:
      * ``label``         : ``"<module>/<experiment>"``
      * ``partial``       : True iff at least one checkpoint family was
                            missing on SendServer (download was incomplete).
      * ``failed_upload`` : True iff any per-file scp upload failed.
      * ``output_text``   : captured log block (parallel mode) or ``""``
                            when ``capture=False`` (sequential mode —
                            output is already on stdout).
    """
    label = f"{module}/{experiment}"
    out: io.StringIO | None = io.StringIO() if capture else None

    print(file=out) if capture else print()
    header(
        f"[{idx}/{total}]  module={module}  experiment={experiment}",
        out=out,
    )

    # --- A.0 PRECHECK (Receive + Send inventory) -------------------------
    sub_header("PRECHECK  (Receive + Send inventory)", out=out)
    recv_exp_dir = f"{recv_base}/{module}/{experiment}"
    send_exp_dir = f"{send_base}/{module}/{experiment}"
    log("INFO", f"probe recv: {recv_user_host}:{recv_exp_dir}", indent=2, out=out)
    existing_remote = remote_list_files(
        remote_dir=recv_exp_dir,
        auth=recv_auth,
        user_host=recv_user_host,
        port=recv_port,
    )
    log(
        "INFO",
        f"receive has {len(existing_remote)} existing file(s)",
        indent=2,
        out=out,
    )

    # Probe Send to know EXACTLY what files should exist on Receive.
    # Without this, we could false-positive skip when Receive has SOME
    # files but Send has MORE (e.g., new checkpoints after prior cascade).
    log("INFO", f"probe send: {send_user_host}:{send_exp_dir}", indent=2, out=out)
    send_files = remote_list_files(
        remote_dir=send_exp_dir,
        auth="",
        user_host=send_user_host,
        port=send_port,
    )
    log(
        "INFO",
        f"send has {len(send_files)} file(s)",
        indent=2,
        out=out,
    )

    # Compute relevant files on Send (checkpoint globs + logs)
    send_relevant: set[str] = set()
    for g in CHECKPOINT_GLOBS:
        for f in send_files:
            if fnmatch.fnmatchcase(f, g):
                send_relevant.add(f)
    send_relevant.update(f for f in send_files if f.startswith(f"{LOGS_DIR}/"))

    if not send_relevant:
        log(
            "SKIP",
            f"{label}: no relevant files on Send; nothing to cascade.",
            indent=2,
            out=out,
        )
        print(file=out) if capture else print()
        log("DONE", f"{label} nothing on Send (no-op)", out=out)
        return label, False, False, (out.getvalue() if capture else "")

    # Set difference: files on Send that are NOT yet on Receive.
    # ONLY skip when this set is EMPTY (Receive has EVERY relevant file).
    missing_from_recv = send_relevant - existing_remote

    if not missing_from_recv:
        log(
            "SKIP",
            f"{label}: all {len(send_relevant)} relevant file(s) from Send "
            f"already on Receive; skip download + upload.",
            indent=2,
            out=out,
        )
        for f in sorted(send_relevant):
            log("SKIP", f"{f}  (already on Receive)", indent=2, out=out)
        print(file=out) if capture else print()
        log("DONE", f"{label} already cascaded (no-op)", out=out)
        return label, False, False, (out.getvalue() if capture else "")

    # Some files need transfer -- log what's missing vs already present
    already_on_recv = send_relevant & existing_remote
    log(
        "INFO",
        f"{len(missing_from_recv)} file(s) missing on Receive, "
        f"{len(already_on_recv)} already present",
        indent=2,
        out=out,
    )
    for f in sorted(missing_from_recv):
        log("NEED", f"{f}  (must transfer)", indent=2, out=out)

    # --- A.1 DOWNLOAD ---------------------------------------------------
    sub_header("DOWNLOAD  (SendServer -> local)", out=out)
    ok, missing = download_experiment(
        module,
        experiment,
        src_user_host=send_user_host,
        src_port=send_port,
        src_base=send_base,
        local_base=local_base,
        ignore_patterns=args.ignore,
        dry_run=args.dry_run,
        out=out,
    )
    partial = bool(missing)
    if missing:
        log("WARN", f"partial download, missing: {missing}", indent=2, out=out)
    if not ok:
        log(
            "SKIP",
            "no artifacts available locally; skip upload, " "move to next experiment.",
            indent=2,
            out=out,
        )
        return label, partial, False, (out.getvalue() if capture else "")

    # --- A.2 UPLOAD -----------------------------------------------------
    sub_header("UPLOAD  (local -> ReceiveServer)", out=out)
    up_ok = upload_experiment(
        module,
        experiment,
        local_base=local_base,
        existing_remote=existing_remote,
        recv_auth=recv_auth,
        recv_user_host=recv_user_host,
        recv_port=recv_port,
        recv_base=recv_base,
        dry_run=args.dry_run,
        out=out,
    )

    # --- A.3 CLEANUP ----------------------------------------------------
    if up_ok:
        sub_header("CLEANUP  (free local disk)", out=out)
        if args.keep_checkpoints:
            log("KEEP", "--keep-checkpoints set; not deleting.", indent=2, out=out)
        else:
            cleanup_checkpoints(
                module,
                experiment,
                local_base=local_base,
                dry_run=args.dry_run,
                out=out,
            )
        print(file=out) if capture else print()
        log("DONE", f"{label} cascaded successfully", out=out)
        return label, partial, False, (out.getvalue() if capture else "")

    print(file=out) if capture else print()
    log(
        "FAIL",
        f"{label} upload failed; local checkpoints PRESERVED for retry.",
        out=out,
    )
    return label, partial, True, (out.getvalue() if capture else "")


# ======================================================================
# Main
# ======================================================================
def main() -> int:
    """Entry point: Send->Local->Receive three-node cascade."""
    args = parse_args()

    # Resolve named server profiles BEFORE validation so the operator
    # gets a single actionable error per missing endpoint (naming both
    # the env key AND the matching CLI override).
    sender_profile = load_server_profile(args.sender)
    receiver_profile = load_server_profile(args.receiver)

    if not args.send_ssh:
        args.send_ssh = profile_ssh(sender_profile)
    if not args.send_storage_root:
        args.send_storage_root = sender_profile["StorageRoot"]
    if not args.receive_ssh:
        args.receive_ssh = profile_ssh(receiver_profile)
    if not args.receive_storage_root:
        args.receive_storage_root = receiver_profile["StorageRoot"]
    if args.password is None:
        args.password = receiver_profile["PWD"] or None

    # Fail-fast: every endpoint must be set, either by the resolved
    # profile or by the matching CLI override. We surface a single
    # actionable error per endpoint so the user knows BOTH ways to fix it.
    if not args.send_ssh:
        raise SystemExit(
            f"[FATAL] Sender SSH endpoint is empty for profile {args.sender!r}. "
            f"Set ``{args.sender}User`` and ``{args.sender}IP`` (or "
            f"``{args.sender}Host``) in ``./.env`` (and optionally "
            f"``{args.sender}Port``), or pass ``--send-ssh 'ssh [-p PORT] "
            "USER@HOST'``."
        )
    if not args.receive_ssh:
        raise SystemExit(
            f"[FATAL] Receiver SSH endpoint is empty for profile {args.receiver!r}. "
            f"Set ``{args.receiver}User`` and ``{args.receiver}IP`` (or "
            f"``{args.receiver}Host``) in ``./.env`` (and optionally "
            f"``{args.receiver}Port``), or pass ``--receive-ssh 'ssh [-p PORT] "
            "USER@HOST'``."
        )
    if not args.send_storage_root:
        raise SystemExit(
            f"[FATAL] Sender storage root is empty for profile {args.sender!r}. "
            f"Set ``{args.sender}StorageRoot`` in ``./.env``, or pass "
            "``--send-storage-root /abs/path``. There is no hardcoded "
            "fallback."
        )
    if not args.receive_storage_root:
        raise SystemExit(
            f"[FATAL] Receiver storage root is empty for profile {args.receiver!r}. "
            f"Set ``{args.receiver}StorageRoot`` in ``./.env``, or pass "
            "``--receive-storage-root /abs/path``. There is no hardcoded "
            "fallback."
        )

    send_port, send_user_host = parse_ssh(args.send_ssh)
    recv_port, recv_user_host = parse_ssh(args.receive_ssh)

    send_base = f"{args.send_storage_root.rstrip('/')}/{EXPERIMENT_SUBPATH}"
    recv_base = f"{args.receive_storage_root.rstrip('/')}/{EXPERIMENT_SUBPATH}"
    local_base = Path(args.local_base).expanduser()

    recv_auth = build_auth_prefix(args.password)
    auth_mode = "password (sshpass)" if args.password else "plain ssh/scp"

    # Resolve modules to process.
    if args.module == ALL_KEYWORD:
        modules = list(VALID_MODULES)
    else:
        modules = [args.module]

    # Resolve datasets (Loss_prepare pass) — default 'all' when -d omitted.
    if args.dataset is None:
        ds_arg: list[str] = [ALL_KEYWORD]
    else:
        ds_arg = list(args.dataset)

    header(
        "LabToLocalToCompusServer  --  SendServer -> LOCAL -> ReceiveServer "
        "(any two hosts via .env)"
    )
    log("INFO", f"sender profile    : {args.sender}")
    log("INFO", f"receiver profile  : {args.receiver}")
    log("INFO", f"SendServer    (DL) : {send_user_host}{port_suffix(send_port)}")
    log("INFO", f"  send_base        : {send_base}")
    log("INFO", f"ReceiveServer (UP) : {recv_user_host}{port_suffix(recv_port)}")
    log("INFO", f"  recv_base     : {recv_base}")
    log("INFO", f"local_base        : {local_base.resolve()}")
    log("INFO", f"modules           : {modules}")
    log("INFO", f"experiment        : {args.experiment}")
    log("INFO", f"datasets          : {ds_arg}")
    log("INFO", f"ignore            : {args.ignore}")
    log("INFO", f"keep_ckpts        : {args.keep_checkpoints}")
    log("INFO", f"auth (receive)     : {auth_mode}")
    log("INFO", f"jobs              : {args.jobs}")
    log("INFO", f"dry_run           : {args.dry_run}")
    hr()
    overall_partial: list[str] = []
    overall_failed_upload: list[str] = []

    for module in modules:
        # ---- A. Experiment pass ------------------------------------------
        if args.experiment == ALL_KEYWORD:
            experiments = discover_experiments(module)
            if not experiments:
                print()
                log(
                    "WARN",
                    f"no experiments discovered for module={module} "
                    f"under {DEFAULT_CONFIGS_ROOT}",
                )
            else:
                print()
                log(
                    "INFO",
                    f"module={module}: {len(experiments)} experiment(s) to cascade",
                )
        else:
            experiments = [args.experiment]
            print()
            log("INFO", f"module={module}: single experiment '{args.experiment}'")

        # Resolve worker count for this module's experiments.
        n_exp = len(experiments)
        jobs = _resolve_jobs(args.jobs, n_exp)
        if n_exp:
            mode = "sequential" if jobs == 1 else f"parallel x {jobs}"
            log("INFO", f"module={module}: cascade mode = {mode}")

        # Common kwargs threaded into every worker invocation.
        worker_kwargs = dict(
            args=args,
            send_user_host=send_user_host,
            send_port=send_port,
            send_base=send_base,
            recv_auth=recv_auth,
            recv_user_host=recv_user_host,
            recv_port=recv_port,
            recv_base=recv_base,
            local_base=local_base,
        )

        if jobs <= 1 or n_exp <= 1:
            # Sequential path: stream output live to the terminal as before.
            for idx, exp in enumerate(experiments, start=1):
                label, partial, failed_up, _ = _cascade_one_experiment(
                    idx, n_exp, module, exp, capture=False, **worker_kwargs
                )
                if partial:
                    overall_partial.append(label)
                if failed_up:
                    overall_failed_upload.append(label)
        else:
            # Parallel path: each worker captures its block of output to a
            # buffer; we flush whole blocks atomically as workers finish.
            with ThreadPoolExecutor(max_workers=jobs) as pool:
                futures = {
                    pool.submit(
                        _cascade_one_experiment,
                        idx,
                        n_exp,
                        module,
                        exp,
                        capture=True,
                        **worker_kwargs,
                    ): (idx, exp)
                    for idx, exp in enumerate(experiments, start=1)
                }
                for fut in as_completed(futures):
                    idx, exp = futures[fut]
                    try:
                        label, partial, failed_up, output = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        label = f"{module}/{exp}"
                        with _OUTPUT_LOCK:
                            print()
                            log(
                                "FAIL",
                                f"{label} worker raised: {exc!r}",
                            )
                            sys.stdout.flush()
                        overall_failed_upload.append(label)
                        continue
                    with _OUTPUT_LOCK:
                        sys.stdout.write(output)
                        sys.stdout.flush()
                    if partial:
                        overall_partial.append(label)
                    if failed_up:
                        overall_failed_upload.append(label)

        # ---- B. Loss_prepare pass ---------------------------------------
        # Skip pass when user explicitly cleared datasets via '-d ""'.
        effective_ds = [d for d in ds_arg if d != ""]
        if not effective_ds:
            print()
            log("SKIP", f"module={module}: Loss_prepare pass skipped (-d '')")
            continue

        if ALL_KEYWORD in effective_ds:
            datasets = discover_datasets(module)
            if not datasets:
                print()
                log(
                    "WARN",
                    f"no datasets discovered for module={module} "
                    f"under {DEFAULT_CONFIGS_ROOT}",
                )
                continue
        else:
            seen: set[str] = set()
            datasets = []
            for d in effective_ds:
                if d not in seen:
                    seen.add(d)
                    datasets.append(d)

        print()
        log(
            "INFO",
            f"module={module}: Loss_prepare pass over "
            f"{len(datasets)} dataset(s): {datasets}",
        )
        for ds in datasets:
            ok = cascade_loss_prepare(
                module,
                ds,
                src_user_host=send_user_host,
                src_port=send_port,
                src_base=send_base,
                local_base=local_base,
                recv_auth=recv_auth,
                recv_user_host=recv_user_host,
                recv_port=recv_port,
                recv_base=recv_base,
                dry_run=args.dry_run,
            )
            if not ok:
                overall_failed_upload.append(f"{module}/{ds}{LOSS_PREPARE_SUFFIX}")

    # ------------------------------------------------------------------
    print()
    header("SUMMARY")
    log("INFO", f"partial downloads : {len(overall_partial)}")
    for x in overall_partial:
        log("WARN", f"  - {x}")
    log("INFO", f"failed uploads    : {len(overall_failed_upload)}")
    for x in overall_failed_upload:
        log("FAIL", f"  - {x}")
    if not overall_partial and not overall_failed_upload:
        log("DONE", "all items cascaded successfully")
    hr()

    return 0 if not overall_failed_upload else 1


if __name__ == "__main__":
    sys.exit(main())
