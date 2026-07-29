#!/usr/bin/env python3
"""Direct cascade: SendServer -> ReceiveServer (run ON the SendServer).

Arguments (every flag must be explicit on the CLI -- no hidden defaults):
    --sender PROFILE          REQUIRED. .env profile for the SOURCE host (local).
    --receiver PROFILE        REQUIRED. .env profile for the DESTINATION host.
    -m METHOD/MODULE          Research method + module (e.g. lcp/builder, lcp/all).
    -e EXPERIMENT             Experiment dir name, or 'all' to auto-discover.
    -d DATASET                Dataset for Loss_prepare pass (repeatable; 'all' default).
    -p PASSWORD               Receive SSH password override (profile PWD if omitted).
    -i PATTERN                Artifact pattern to skip (repeatable).
    -j N                      Parallel worker count ('auto' default).
    --send-storage-root PATH  Override sender StorageRoot from .env.
    --receive-ssh SSH_STR     Override receiver SSH string from .env.
    --receive-storage-root PATH  Override receiver StorageRoot from .env.
    --dry-run                 Print commands; touch no files.

Effects:
    Sender   : resolved from ./.env[<sender>StorageRoot] (LOCAL filesystem)
    Receiver : resolved from ./.env[<receiver>{User,Host,Port,PWD,StorageRoot}]
    Transfer : content-aware delta (only missing files shipped)
    Parallel : experiments cascade concurrently (per -j)
    Idempotent : re-runs skip files already on Receive

Usage::

    python3 LabToCampusServer.py --sender LabServer --receiver CampusServer -p '...' --dry-run
    python3 LabToCampusServer.py --sender LabServer --receiver CampusServer -p '...' -m lcp/builder -e GSM8K_Qwen2.5-0.5B_4level
    python3 LabToCampusServer.py --sender LabServer --receiver CampusServer -p '...' -j 1

----------------------------------------------------------------------
Detailed description
----------------------------------------------------------------------
Run this script DIRECTLY on the SendServer. It walks the Send filesystem
LOCALLY (no ssh into Send), then ships missing artifacts to Receive via a
single ``scp`` hop (Send -> Receive).

+----------------+-----------------------------------------------------+
| Role           | Sender profile (SOURCE host == LOCAL host)          |
| Selected by    | ``--sender <Profile>`` (REQUIRED)                   |
| Storage root   | $<Sender>StorageRoot  (read from ``./.env``)        |
| Full src path  | <SenderStorageRoot>/<method>/<module>/<experiment>/ |
+----------------+-----------------------------------------------------+
| Role           | Receiver profile (DESTINATION host)                 |
| Selected by    | ``--receiver <Profile>`` (REQUIRED)                 |
| Endpoint       | $<Receiver>User @ $<Receiver>{Host|IP} [: $<Receiver>Port] |
| Storage root   | $<Receiver>StorageRoot  (read from ``./.env``)      |
| Full tgt path  | <ReceiverStorageRoot>/<method>/<module>/<experiment>/ |
+----------------+-----------------------------------------------------+

IMPORTANT: ``EXPERIMENT/`` is part of the user-supplied storage root
(not auto-appended). The script joins ONLY ``/<method>`` after the
root, so you control the entire prefix. Pass ``--send-storage-root``
/ ``--receive-storage-root`` on the CLI to override the values from
``./.env``. There is NO hardcoded fallback for any path -- if a key
is unset in ``./.env`` AND the matching CLI flag is not passed, the
script fails fast at startup with an actionable error message.

The ``<method>`` segment identifies which research method (folder under
``configs/``) is being cascaded; it defaults to ``lcp`` and is
overridden via ``-m <method>/<module>`` (e.g. ``-m nlcp/builder``).
Both servers SHARE the exact relative path
``<method>/<module>/<experiment>/`` (encoded in every config's
``log.save_folder``) APPENDED to whichever storage root each side
uses; only the storage ROOT differs. This is what keeps the SAME
config files usable on Receive right after the cascade. Discovery
uses the Send-local ``configs/<method>/**/train_<module>_*.yml``
files as the source of truth (the repo must be checked out on the
SendServer).

Pipeline (run-wide; per-experiment work parallelised via ``-j/--jobs``):

    PHASE 1  INVENTORY (both sides, by module)
        * Send     : LOCAL filesystem walk of
                    ``<send_base>/<module>/<exp>/`` for every <exp>
                    found on disk. ``<send_base>`` is
                    ``<send_storage_root>/<method>``.
        * Receive  : ONE ssh per module (``find <recv_base>/<module>``)
                    that lists every file under every experiment dir.
        For both sides we keep ONLY the files we actually cascade --
        i.e. files matching ``CHECKPOINT_GLOBS`` plus everything under
        ``logs/``. "Has content" means "has at least one such relevant
        file"; an empty experiment dir is treated as absent.

    PHASE 2  DIFF
        For every Send experiment with content, compare its relevant-file
        set to the Receive relevant-file set:
          * If ``send_relevant - recv_relevant`` is EMPTY -> [SKIP]:
            Receive already has every artifact, no transfer is queued.
          * Otherwise -> [QUEUE]: print the count of missing files and
            add the experiment to the parallel transfer queue.
        Send dirs with no relevant content are reported as [NO-SEND]
        and never queued (they are NOT errors).

    PHASE 3  PARALLEL TRANSFER
        ``ThreadPoolExecutor(max_workers=jobs)`` runs one worker per
        queued experiment. Each worker:
          * mkdir -p the parent dirs on Receive (single ssh per worker),
          * scp Send -> Receive for every missing file,
          * captures the FULL per-file log into a JSON record at
            ``<send_storage_root>/SendToReceive/``
            ``<method>__<module>__<experiment>.json``.
        The TERMINAL only sees one short line per finished worker:
          ``[OK]   [3/12] lcp/builder/GSM8K_xxx  in 12.3s -> <json_path>``
        Anything verbose (per-file scp commands, mkdir output, missing
        glob warnings) lives in the JSON file.

    PHASE 4  LOSS_PREPARE PASS (per dataset, sequential)
        For each ``<dataset>_Loss_prepare.json`` declared in
        ``configs/<method>/<dataset>/.../train_<module>_*.yml``: skip
        if already on Receive, else mkdir -p + scp.

    PHASE 5  SUMMARY
        Run-wide totals: scanned, skipped (already on Receive),
        queued, succeeded, failed; plus the path to the JSON log dir.

Pre-requisites on the SendServer (where this script runs):
    * ``sshpass`` available when Receive uses password auth (-p). If
      Send has SSH-key trust to Receive, omit ``-p`` and plain
      ssh/scp is used.
    * Network reachability Send -> Receive on the receive port.

Failure policy:
    * If SendServer has no files matching a CHECKPOINT_GLOBS pattern,
      that pattern is reported [PARTIAL]; we still attempt every
      other queued file for that experiment.
    * If a transfer fails for any file, the experiment is marked
      [FAIL] but the next experiment continues.

----------------------------------------------------------------------
WHAT IS DISCOVERED  (the local configs are the SINGLE source of truth)
----------------------------------------------------------------------
Nothing about the cascade is hard-coded -- it is fully driven by the
YAML configs checked out on the SendServer at
``./configs/<method>/`` (default method: ``lcp``).

Directory convention:
    configs/<method>/<dataset>/[<sub>/]train_<module>_*.yml
        <method>  : research method name = top-level folder under
                    ``configs/`` (e.g. ``lcp``, ``nlcp``). Selected
                    via the optional method prefix in ``-m``;
                    defaults to ``lcp``.
        <dataset> : FIRST path component under ``configs/<method>/``
                    (e.g. ``GSM8K``, ``MATH``).
        <module>  : ``builder`` OR ``predictor`` (chosen via -m).
        *.yml     : any number of training configs per dataset/module.

For each YAML the script reads ``log.save_folder`` and uses its
``Path(...).name`` as the experiment name. That same name is the
leaf directory of every artifact dir on BOTH servers.

Example: with default method ``lcp`` and a config containing
    log:
      save_folder: ./EXPERIMENT/lcp/builder/GSM8K_Qwen2.5-0.5B_6level
yields experiment name ``GSM8K_Qwen2.5-0.5B_6level``, which the
trainer materialises as a directory on Send at
    /Data/ReasoningNLCP/EXPERIMENT/lcp/builder/GSM8K_Qwen2.5-0.5B_6level/
and the cascade mirrors to Receive at
    /data/user/<ReceiverUser>/ReasoningNLCP/EXPERIMENT/lcp/builder/GSM8K_Qwen2.5-0.5B_6level/
For a different method (e.g. ``nlcp``), substitute ``lcp`` with
``nlcp`` everywhere above and invoke ``-m nlcp/<module>``.

CLI selectors:
    (no -m given)           method = ``lcp`` (DEFAULT_METHOD),
                            module = ``all`` (ALL_KEYWORD).
                            This is the ONLY case where -m may be
                            omitted -- a missing -m is the ONLY
                            implicit form.
    -m <method>/<module>    REQUIRED full form whenever -m is set.
                            NO defaulting is performed: bare values
                            like ``-m builder`` / ``-m all`` /
                            ``-m nlcp`` are REJECTED.
                            <method> = top-level folder under
                                       ``configs/`` (e.g. ``lcp``,
                                       ``nlcp``); also the per-method
                                       subdir appended to each
                                       storage root.
                            <module> in {builder, predictor, all}.
                            Examples: ``lcp/all``, ``lcp/builder``,
                                      ``nlcp/all``, ``nlcp/predictor``.
    -e all                  take EVERY experiment found via YAML sweep
    -e <name>               transfer ONLY the experiment dir of that
                            exact name (still under
                            EXPERIMENT/<method>/<module>/, no YAML
                            parsing in this case)
    -d all                  (Loss_prepare pass) datasets = first path
                            component of every matching
                            ``configs/<method>/<ds>/.../train_<module>_*.yml``
    -d <ds>                 repeat to limit; use multiple ``-d`` for
                            several datasets
    -d ''                   empty string -> SKIP the Loss_prepare pass

----------------------------------------------------------------------
WHAT IS TRANSFERRED  (per experiment + per dataset)
----------------------------------------------------------------------
For EACH discovered experiment ``<module>/<experiment>``, the cascade
ships -- relative to ``<storage_root>/<method>/<module>/<experiment>/``
(where ``<storage_root>`` is the user-supplied root that ALREADY
includes the ``EXPERIMENT/`` segment) -- the following files (and
ONLY these), all under their original relative subpaths:

    A. Best checkpoints (CHECKPOINT_GLOBS):
         checkpoints/checkpoint_best_eval-*.pt   (best by eval metric)
         checkpoints/checkpoint_best-*.pt        (best by train loss)
       The trainer encodes ``-epoch<E>-step<S>.pt`` suffixes; both
       globs match independently. NON-best snapshots
       (e.g. ``checkpoint_latest-*.pt``, ``checkpoint_epoch*.pt``)
       are NOT shipped -- this script is intentionally restricted to
       the two ``best*`` families to keep Receive disk usage small.

    B. The ENTIRE ``logs/`` subtree:
         logs/**/*
       Loss curves, TensorBoard event files, console captures,
       per-epoch metric JSONs -- every file the trainer wrote under
       ``logs/`` is mirrored verbatim, preserving the relative layout.

For EACH selected dataset (Loss_prepare pass), the cascade ships
ONE small JSON file at ``<storage_root>/<method>/<module>/`` (note:
SIBLING of the experiment dirs, not nested inside one):

    C. <dataset>_Loss_prepare.json
       Per-sample teacher-forcing loss/perplexity table emitted by
       the loss-prepare pipeline; consumed by Receive-side analysis.

NOT transferred (intentionally):
    * Any file under the experiment dir whose basename does NOT match
      a CHECKPOINT_GLOBS pattern AND is not under ``logs/``
      (e.g. intermediate ``.tmp`` files, manually placed scratch).
    * The local ``configs/<method>/`` tree itself -- configs are part
      of the repo and are pushed via git, not via this cascade.
    * The ``raw_data/``, ``cache/`` and similar working dirs.
    * Anything whose relative path matches a ``-i`` ignore pattern
      (exact / parent-dir / fnmatch glob, repeatable).

Idempotent skips (no overwrite ever):
    * Per-file: each Send artifact is shipped only when its relative
      path is NOT already in the Receive inventory.
    * Per-experiment: when EVERY relevant Send artifact (best ckpts +
      logs/) is already mirrored on Receive, the whole experiment is
      tagged [SKIP] -> [DONE no-op] without launching any scp.
    * Per-Loss_prepare: a single ``ls`` probe on Receive skips the
      whole upload when the JSON is already there.

----------------------------------------------------------------------
WHERE PER-EXPERIMENT JSON LOGS GO  (so the terminal can stay quiet)
----------------------------------------------------------------------
Every Phase-3 worker writes its FULL per-file log into a JSON file.
The terminal only sees one short line per finished worker; everything
verbose (per-file scp commands, batched ``mkdir -p``, missing-glob
warnings, raw stderr) lives in the JSON file. The location is:

    <send_storage_root>/SendToReceive/
        <method>__<module>__<experiment>.json

where
    <send_storage_root>  = the value of ``--send-storage-root`` (default:
                          ``/Data/ReasoningNLCP/EXPERIMENT``).
    SendToReceive  fixed name (``LOG_OUTPUT_SUBDIR``).
    <method>__<module>__<experiment>.json
                              ONE file per experiment that ENTERED
                              Phase 3 (i.e. queued for transfer).
                              Re-runs OVERWRITE the same file (no
                              timestamped subdirs -- keeps the layout
                              flat and predictable). Skipped / no-Send
                              experiments are NOT materialised as JSON;
                              they are only counted in the Phase-2 /
                              Phase-5 console summaries.

Concrete example (defaults + ``-m lcp/builder``):

    /Data/ReasoningNLCP/EXPERIMENT/SendToReceive/
        lcp__builder__GSM8K_Qwen2.5-0.5B_3level_AutoWeighted.json
        lcp__builder__GSM8K_Qwen2.5-0.5B_6level.json
        ...

The full path of every JSON written is also echoed live in the
terminal next to its ``[OK]`` / ``[FAIL]`` line, AND once more in the
final SUMMARY block (``json log dir : ...``). Each JSON record carries
at minimum:

    method, module, experiment
    send_path, recv_path
    started_at / finished_at / duration_seconds
    send_files_total, send_files_relevant, recv_files_existing
    files_to_transfer, pending_files (full list)
    missing_globs, dry_run, result (OK | FAIL | ERROR), error?
    log  -- the full per-file ``[INFO]/[XFR]/[OK]/[FAIL]`` lines that
            would otherwise have been printed to the terminal.

----------------------------------------------------------------------
PRE-FLIGHT SETUP  (do this ONCE per machine; cascade is .env-driven)
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
      experiment-name leaf that BOTH servers share underneath their
      respective storage roots, which is what makes the cascaded
      checkpoints immediately re-trainable on Receive with the SAME
      config files.

Step 1 -- Configure ``./.env`` with NAMED SERVER PROFILES. Each
profile is a flat ``<Profile><Attr>`` namespace; only the attributes
relevant to a given role are consumed. For this script the SENDER
profile contributes ``StorageRoot`` and the RECEIVER profile
contributes ``User`` / ``Host`` (or ``IP``) / ``Port`` / ``PWD`` /
``StorageRoot``. Example minimal layout (see ``./.env`` for the live
template)::

    # Sender profile (SOURCE of artifacts; this script runs HERE)
    LabServerIP             = 10.123.4.30
    LabServerUser           = sjia
    LabServerStorageRoot    = /Data/ReasoningNLCP/EXPERIMENT

    # Receiver profile (DESTINATION of artifacts)
    CampusServerIP          = 10.120.48.27
    CampusServerUser        = alice
    CampusServerHost        = hpcfront.example.edu        # OPTIONAL; falls
                                                          # back to <P>IP
    CampusServerPort        =                             # OPTIONAL; empty ->
                                                          # ssh client default
    CampusServerPWD         = <password>                  # OPTIONAL; when set,
                                                          # sshpass is invoked
                                                          # for non-interactive
                                                          # auth on every ssh /
                                                          # scp call. Omit when
                                                          # SSH-key trust is
                                                          # already configured.
    CampusServerStorageRoot = /data/user/alice/ReasoningNLCP/EXPERIMENT

    # Optional ergonomics: argument-free invocation.
    DefaultSender   = LabServer
    DefaultReceiver = CampusServer

At startup the script resolves the chosen profiles into the four
transfer parameters that show up in the banner::

    sender                    -> ``--sender`` or ``$DefaultSender``
    receiver                  -> ``--receiver`` or ``$DefaultReceiver``
    send_storage_root         -> ``$<Sender>StorageRoot``
    receive_ssh / password    -> from ``$<Receiver>{User,Host|IP,Port,PWD}``
    receive_storage_root      -> ``$<Receiver>StorageRoot``

Any CLI flag (``--sender`` / ``--receiver`` / ``--receive-ssh`` /
``--send-storage-root`` / ``--receive-storage-root`` / ``-p``) takes
precedence; the ``.env`` values only seed the defaults so the normal
invocation can be argument-free::

    python3 LabToCampusServer.py        # uses .env profiles transparently
    python3 LabToCampusServer.py --sender LabServer --receiver CampusServer

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
``<SenderStorageRoot>/<method>/<module>/<experiment-name>/`` and that
this script mirrors verbatim onto the Receiver at
``<ReceiverStorageRoot>/<method>/<module>/<experiment-name>/``.

Step 3 -- Run the cascade::

    python3 LabToCampusServer.py        # full sweep, .env defaults

----------------------------------------------------------------------
Usage (run ON the SendServer)
----------------------------------------------------------------------
With ``./.env`` configured, NO arguments are required at all. Pass
``-p`` only when overriding the env-supplied password (or when the
``.env`` is intentionally minimal).

    # Default sweep: NO -m given -> method=lcp, module=all.
    # send_storage = $<Sender>StorageRoot from ./.env,
    # recv_storage = $<Receiver>StorageRoot from ./.env.
    # Discovers EVERY ``configs/lcp/**/train_builder_*.yml`` AND
    # ``configs/lcp/**/train_predictor_*.yml``, cascades for each:
    #   <send_storage>/lcp/<module>/<exp>/checkpoints/checkpoint_best-*.pt
    #   <send_storage>/lcp/<module>/<exp>/checkpoints/checkpoint_best_eval-*.pt
    #   <send_storage>/lcp/<module>/<exp>/logs/**
    # then sweep every dataset's <dataset>_Loss_prepare.json. Workers
    # run in parallel (``-j auto`` -> min(4, n_experiments)).
    python3 LabToCampusServer.py -p 'YourCampusPassword'

    # Restrict to ONE module. Whenever -m is given, the FULL
    # <method>/<module> form is REQUIRED -- bare ``-m builder`` /
    # ``-m all`` / ``-m predictor`` are REJECTED.
    python3 LabToCampusServer.py -m lcp/all        -p '...'
    python3 LabToCampusServer.py -m lcp/builder    -p '...'
    python3 LabToCampusServer.py -m lcp/predictor  -p '...'

    # Target a DIFFERENT method (e.g. ``nlcp``). Files are discovered
    # under ``configs/nlcp/`` and cascaded under
    # ``<storage_root>/nlcp/<module>/<exp>/``.
    python3 LabToCampusServer.py -m nlcp/all       -p '...'
    python3 LabToCampusServer.py -m nlcp/builder   -p '...'
    python3 LabToCampusServer.py -m nlcp/predictor -p '...'

    # ONE specific experiment, skipping YAML auto-discovery. The
    # experiment dir must already exist at
    # <send_storage_root>/<method>/<module>/<exp>/, e.g.
    # /Data/ReasoningNLCP/EXPERIMENT/lcp/builder/GSM8K_Qwen2.5-0.5B_6level/.
    python3 LabToCampusServer.py -m lcp/builder  -e GSM8K_Qwen2.5-0.5B_6level -p '...'
    python3 LabToCampusServer.py -m nlcp/builder -e GSM8K_Qwen2.5-0.5B_6level -p '...'

    # Limit the Loss_prepare pass to specific datasets (repeatable).
    python3 LabToCampusServer.py -p '...' -d GSM8K
    python3 LabToCampusServer.py -p '...' -d GSM8K -d MATH

    # Skip the Loss_prepare pass entirely (only ckpts + logs/).
    python3 LabToCampusServer.py -p '...' -d ''

    # Skip a checkpoint family per experiment via ``-i`` (repeatable,
    # supports exact / parent-dir / fnmatch glob).
    python3 LabToCampusServer.py -p '...' -i 'checkpoints/checkpoint_best-*.pt'
    python3 LabToCampusServer.py -p '...' -i logs    # skip the whole logs/ subtree
    python3 LabToCampusServer.py -p '...' -i 'checkpoints/*best.pt' -i logs

    # Override storage roots from ``./.env`` (the root is taken AS-IS;
    # the script only appends ``/<method>``). Include the
    # ``EXPERIMENT/`` segment yourself, or any parent layout you
    # prefer. NO hardcoded fallback -- one of these MUST resolve
    # (env or CLI) for each side.
    python3 LabToCampusServer.py -p '...' --send-storage-root /Data2/ReasoningNLCP/EXPERIMENT
    python3 LabToCampusServer.py -p '...' --receive-storage-root /data/user/other/ReasoningNLCP/EXPERIMENT
    python3 LabToCampusServer.py -p '...' --send-storage-root /Data/CustomLayout/RUNS

    # Override the Receive SSH endpoint (e.g. switch HPC frontends).
    python3 LabToCampusServer.py -p '...' --receive-ssh 'ssh -p 22 <user>@<recv_host>'

    # Strict sequential mode (live scp output to terminal, one
    # experiment at a time -- easiest to inspect / debug).
    python3 LabToCampusServer.py -p '...' -j 1

    # Crank parallel transfers up (each worker = one experiment
    # end-to-end; logs are buffered per-worker and flushed atomically).
    python3 LabToCampusServer.py -p '...' -j 8

    # Dry-run: print EVERY ssh / scp command the cascade WOULD issue
    # (each prefixed with ``[DRY] $``) but execute none. Combine with
    # the selectors above to preview a real run safely.
    python3 LabToCampusServer.py -p '...' --dry-run
    python3 LabToCampusServer.py -m lcp/predictor -e MATH_Qwen2.5-0.5B_4level -p '...' --dry-run
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import TextIO

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


# Default research method. ``-m <method>/<module>`` overrides; bare
# ``-m <module>`` keeps this default. Drives BOTH the configs root
# (``./configs/<method>``) and the per-method subdir appended to each
# storage root (``<storage_root>/<method>``).
DEFAULT_METHOD = "lcp"
CONFIGS_BASE = Path("./configs")

# Globs are matched LOCALLY on Send against the filesystem walk
# (``Path.rglob``-derived inventory) using ``fnmatch.fnmatchcase``.
CHECKPOINT_GLOBS = [
    "checkpoints/checkpoint_best_eval-*.pt",
    "checkpoints/checkpoint_best-*.pt",
]
LOGS_DIR = "logs"
LOSS_PREPARE_SUFFIX = "_Loss_prepare.json"

# Subdir under <send_storage_root>/ where per-experiment JSON transfer
# logs are written. Layout is FLAT -- one JSON file per transferred
# experiment named ``<method>__<module>__<experiment>.json`` directly
# inside the subdir. Re-runs overwrite the same file (no timestamped
# nesting). Terminal output stays brief; the JSON files contain the
# full per-file log.
LOG_OUTPUT_SUBDIR = "SendToReceive"

VALID_MODULES = ("builder", "predictor")
ALL_KEYWORD = "all"

# ----------------------------------------------------------------------
# Display helpers.
# Tags used here:
#   [ XFR]   -- direct Send -> Receive scp transfer
#   [SKIP]   -- artifact already at destination / nothing to do
#   [ OK ]   -- step succeeded
#   [DONE]   -- end-to-end cascade succeeded for one item
#   [MISS]   -- source artifact not found on SendServer
#   [WARN]   -- non-fatal anomaly (continue)
#   [FAIL]   -- step failed (stop the current item, continue with next)
#   [INFO]   -- neutral status / configuration line
# ----------------------------------------------------------------------
LINE_W = 78
TAG_W = 6

# Same lock-and-buffer trick as LabToLocalToCompusServer.py: workers in
# parallel mode write to per-thread StringIO buffers, then flush whole
# blocks under this lock so log lines never interleave on stdout.
_OUTPUT_LOCK = threading.Lock()


def _tag(tag: str) -> str:
    return f"[{tag.center(TAG_W - 2)}]"


def log(tag: str, msg: str, *, indent: int = 0, out: TextIO | None = None) -> None:
    target = out if out is not None else sys.stdout
    print(f"{' ' * indent}{_tag(tag)} {msg}", file=target)


def hr(char: str = "=", *, out: TextIO | None = None) -> None:
    target = out if out is not None else sys.stdout
    print(char * LINE_W, file=target)


def header(title: str, char: str = "=", *, out: TextIO | None = None) -> None:
    target = out if out is not None else sys.stdout
    hr(char, out=target)
    print(title, file=target)
    hr(char, out=target)


def sub_header(title: str, *, out: TextIO | None = None) -> None:
    target = out if out is not None else sys.stdout
    print(file=target)
    print("-" * LINE_W, file=target)
    print(f"  {title}", file=target)
    print("-" * LINE_W, file=target)


# ======================================================================
# CLI parsing
# ======================================================================
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for direct Send->Receive cascade."""
    parser = argparse.ArgumentParser(
        description=(
            "Direct artifact cascade SendServer -> ReceiveServer over a "
            "single scp hop. Run this script ON the SendServer. "
            "Both endpoints are NAMED SERVER PROFILES in ``./.env`` "
            "(``--sender <Profile>`` / ``--receiver <Profile>``); every "
            "flag below is an inline override. Works for ANY two hosts "
            "-- no source-code edit needed to add a new profile."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sender",
        default="",
        metavar="PROFILE",
        help=(
            "Named server profile (from ``./.env``) describing the "
            "DOWNLOAD source -- this script runs ON sender, so only "
            "``<P>StorageRoot`` is consumed (used as --send-storage-root "
            "default). REQUIRED."
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
            "Receive SSH login password. Used by sshpass for both "
            "(a) the Receive PRECHECK ssh probe and (b) the Send->Receive "
            "scp. Default: ``--receiver`` profile key ``<P>PWD``. Pass "
            "``-p ''`` to force key-based auth and ignore the profile "
            "value."
        ),
    )
    parser.add_argument(
        "-m",
        "--module",
        default=None,
        metavar="METHOD/MODULE",
        help=(
            f"Research method + module to process. REQUIRED full "
            f"form whenever this flag is given: '<method>/<module>' "
            f"(e.g. '{DEFAULT_METHOD}/builder', "
            f"'{DEFAULT_METHOD}/{ALL_KEYWORD}', 'nlcp/{ALL_KEYWORD}'). "
            f"<module> must be one of {{builder, predictor, "
            f"{ALL_KEYWORD}}}. Bare values like '-m builder' / "
            f"'-m {ALL_KEYWORD}' are REJECTED. The chosen <method> "
            f"drives BOTH the configs sweep root "
            f"(``./configs/<method>``) and the per-method subdir "
            f"appended to each storage root "
            f"(``<storage_root>/<method>``). OMIT this flag entirely "
            f"to use the default '{DEFAULT_METHOD}/{ALL_KEYWORD}'."
        ),
    )
    parser.add_argument(
        "-e",
        "--experiment",
        default=ALL_KEYWORD,
        help=(
            "Experiment dir name under EXPERIMENT/<method>/<module>/, or "
            "'all' to auto-discover from "
            "configs/<method>/**/train_<module>_*.yml."
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
        "--send-storage-root",
        default="",
        help=(
            "SendServer storage root (LOCAL absolute path). Taken AS-IS "
            "-- the send base becomes ``<root>/<method>``. Inline "
            "override; when empty, read from the --sender profile's "
            "``StorageRoot`` attribute."
        ),
    )
    parser.add_argument(
        "--receive-ssh",
        default="",
        help=(
            "ReceiveServer SSH string of the form ``ssh -p <port> <user>@<host>``. "
            "Inline override; when empty, built from the --receiver "
            "profile's ``User`` / ``Host`` / ``Port`` attributes."
        ),
    )
    parser.add_argument(
        "--receive-storage-root",
        default="",
        help=(
            "ReceiveServer storage root. Taken AS-IS -- the receive base "
            "becomes ``<root>/<method>``. Use the SAME final layout as "
            "the send side so configs round-trip unchanged. Inline "
            "override; when empty, read from the --receiver profile's "
            "``StorageRoot`` attribute."
        ),
    )
    parser.add_argument(
        "-i",
        "--ignore",
        action="append",
        default=[],
        metavar="PATTERN",
        help=(
            "Artifact pattern to SKIP, relative to the experiment dir "
            "(exact / parent / fnmatch glob). Repeatable. "
            "Example: '-i checkpoints/*best.pt'."
        ),
    )
    parser.add_argument(
        "-j",
        "--jobs",
        default="auto",
        metavar="N",
        help=(
            "Parallel worker count for the experiment cascade. Each "
            "worker handles one experiment end-to-end (PRECHECK -> "
            "DIRECT TRANSFER). 'auto' (default) picks "
            "min(4, len(experiments)). Pass an int to override "
            "(e.g. '-j 1' = strict sequential, '-j 8' = up to 8 "
            "concurrent experiments)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print every ssh / scp command without executing it.",
    )
    return parser.parse_args()


# ======================================================================
# Tiny helpers
# ======================================================================
def configs_root_for(method: str) -> Path:
    """Return ``./configs/<method>`` -- the Send-local YAML sweep root."""
    return CONFIGS_BASE / method


def parse_module_spec(spec: str) -> tuple[str, str]:
    """Parse the ``-m`` value into ``(method, module)``.

    REQUIRES the full ``<method>/<module>`` form -- once ``-m`` is
    given, NO defaulting is performed. Bare values such as
    ``builder`` / ``predictor`` / ``all`` / ``nlcp`` are REJECTED.
    Omit ``-m`` entirely to fall back to the default
    ``DEFAULT_METHOD / ALL_KEYWORD`` (handled in ``main`` before
    this function is called).

    Examples (accepted):
      * ``lcp/all``, ``lcp/builder``, ``lcp/predictor``
      * ``nlcp/all``, ``nlcp/builder``, ``nlcp/predictor``
    """
    if "/" not in spec:
        raise SystemExit(
            f"Invalid -m '{spec}': must be the full "
            f"'<method>/<module>' form "
            f"(e.g. '{DEFAULT_METHOD}/builder', "
            f"'{DEFAULT_METHOD}/{ALL_KEYWORD}', 'nlcp/builder'). "
            f"Bare values like 'builder' or '{ALL_KEYWORD}' are no "
            f"longer accepted -- omit -m entirely to use the "
            f"default '{DEFAULT_METHOD}/{ALL_KEYWORD}'."
        )
    method, _, module = spec.partition("/")
    method = method.strip()
    module = module.strip()
    if not method or not module:
        raise SystemExit(
            f"Invalid -m '{spec}': use '<method>/<module>' "
            f"(e.g. '{DEFAULT_METHOD}/builder' or "
            f"'{DEFAULT_METHOD}/{ALL_KEYWORD}')."
        )
    if module not in (*VALID_MODULES, ALL_KEYWORD):
        valid = list(VALID_MODULES) + [ALL_KEYWORD]
        raise SystemExit(
            f"Invalid -m '{spec}': module part must be one of {valid} "
            f"(got '{module}')."
        )
    return method, module


# ======================================================================
# Tiny SSH / auth helpers
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
    """Probe Receive via ssh; True iff ``ls -1 <glob_path>`` lists at
    least one entry. Read-only; runs unconditionally (incl. dry-run).
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
    """Recursive file inventory of ``remote_dir`` on Receive.

    Returns a set of POSIX paths RELATIVE to ``remote_dir`` (no
    leading ``./``). Empty set if the directory does not exist on
    the remote (silently). Read-only; runs unconditionally (incl.
    dry-run) because the result drives delta-transfer decisions.
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


def local_list_files(local_dir: Path) -> set[str]:
    """Recursive file inventory of ``local_dir`` on the SendServer.

    Returns a set of POSIX paths RELATIVE to ``local_dir`` (no leading
    ``./``). Empty set if the directory does not exist. The Send-side
    counterpart of ``remote_list_files``.
    """
    if not local_dir.is_dir():
        return set()
    out: set[str] = set()
    for p in local_dir.rglob("*"):
        if p.is_file():
            out.add(p.relative_to(local_dir).as_posix())
    return out


def run(cmd: str, *, dry_run: bool = False, out: TextIO | None = None) -> int:
    """Run a shell command and return its exit code.

    ``out`` selects where the command echo + child output go:
      * ``None`` / ``sys.stdout`` -> stream child output to the terminal
        in real time (sequential mode).
      * a ``StringIO`` -> capture child stdout+stderr and append to the
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
    """Return the first -i pattern matching ``rel_path``, else None."""
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


def build_scp_cmd(
    *,
    send_path: str,
    recv_user_host: str,
    recv_port: str,
    recv_path: str,
    recv_auth: str,
) -> str:
    """Build the direct ``scp`` command Send -> Receive.

    Single-quote both ``send_path`` (already a local path on Send) and
    ``recv_path`` so the local shell does not glob-expand or split.
    """
    return (
        f"{recv_auth}scp{scp_port_flag(recv_port)} '{send_path}' "
        f"{recv_user_host}:'{recv_path}'"
    )


# ======================================================================
# Config discovery (LOCAL on Send -- same source of truth as the staged variant)
# ======================================================================
def discover_experiments(module: str, configs_root: Path) -> list[str]:
    """configs/<method>/**/train_<module>_*.yml -> [experiment_name, ...].

    Raises:
        FileNotFoundError: when ``configs_root`` does not exist or is not
            a directory. Failing loudly here is intentional: a missing
            configs root makes the whole YAML-driven candidate set
            unknowable, and silently returning ``[]`` would let Phase 2
            mis-classify every Send dir as ``no-yaml-anchor``.
        KeyError: when a matched ``train_*.yml`` lacks the required
            ``log.save_folder`` field (used to derive the experiment
            name).
    """
    if not configs_root.is_dir():
        raise FileNotFoundError(
            f"Configs root not found or not a directory: {configs_root}. "
            f"Check ``-m <method>/<module>`` and that ``./configs/<method>`` "
            f"exists in the current repo checkout."
        )
    prefix = f"train_{module}_"
    seen: set[str] = set()
    out: list[str] = []
    for yml in sorted(configs_root.rglob(f"{prefix}*.yml")):
        with yml.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        try:
            save_folder = cfg["log"]["save_folder"]
        except (KeyError, TypeError) as exc:
            raise KeyError(
                f"YAML {yml} is missing required field 'log.save_folder' "
                f"(needed to derive the experiment name); fix the config "
                f"or remove the stale file from {configs_root}."
            ) from exc
        name = Path(save_folder).name
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def discover_datasets(module: str, configs_root: Path) -> list[str]:
    """configs/<method>/<dataset>/[**/]train_<module>_*.yml -> [dataset, ...].

    Raises:
        FileNotFoundError: when ``configs_root`` does not exist or is not
            a directory (same rationale as :func:`discover_experiments`).
    """
    if not configs_root.is_dir():
        raise FileNotFoundError(
            f"Configs root not found or not a directory: {configs_root}. "
            f"Check ``-m <method>/<module>`` and that ``./configs/<method>`` "
            f"exists in the current repo checkout."
        )
    prefix = f"train_{module}_"
    seen: set[str] = set()
    out: list[str] = []
    for yml in sorted(configs_root.rglob(f"{prefix}*.yml")):
        rel = yml.relative_to(configs_root)
        if len(rel.parts) < 2:
            continue
        ds = rel.parts[0]
        if ds in seen:
            continue
        seen.add(ds)
        out.append(ds)
    return out


# ======================================================================
# Per-experiment direct transfer (Send -> Receive, no local stage)
# ======================================================================
def transfer_experiment(
    module: str,
    experiment: str,
    *,
    send_base: str,
    recv_user_host: str,
    recv_port: str,
    recv_base: str,
    recv_auth: str,
    existing_remote: set[str],
    send_inventory: set[str],
    ignore_patterns: list[str],
    dry_run: bool,
    out: TextIO | None = None,
) -> tuple[bool, list[str]]:
    """Direct delta transfer Send -> Receive for one experiment.

    ``send_inventory``     POSIX paths (relative to the Send exp dir) of
                          every file under that dir on Send.
    ``existing_remote``   POSIX paths (relative to the Receive exp dir)
                          of every file already on Receive.

    The delta is the union of (a) Send files matching CHECKPOINT_GLOBS
    and (b) Send files under ``logs/``, MINUS files already mirrored on
    Receive. Each delta file is shipped via a direct ``scp`` from Send
    to Receive.

    Returns ``(ok, missing_globs)``:
      * ok            True iff every queued file transferred
                      successfully (or the queue was empty).
      * missing_globs CHECKPOINT_GLOBS that have NO match on Send.
    """
    send_exp_dir = f"{send_base}/{module}/{experiment}"
    recv_exp_dir = f"{recv_base}/{module}/{experiment}"

    missing_globs: list[str] = []
    # ``candidates`` holds POSIX paths (relative to the Send exp dir) of
    # every file that the cascade plans to push, in queue order.
    candidates: list[str] = []

    # --- Checkpoint families ---------------------------------------
    for glob_pattern in CHECKPOINT_GLOBS:
        hit = _matches_ignore(glob_pattern, ignore_patterns)
        if hit is not None:
            log("SKIP", f"{glob_pattern}  (ignored by -i {hit!r})", indent=2, out=out)
            continue
        send_matches = sorted(
            p for p in send_inventory if fnmatch.fnmatchcase(p, glob_pattern)
        )
        if not send_matches:
            log("MISS", f"{glob_pattern}  (no match on Send)", indent=2, out=out)
            missing_globs.append(glob_pattern)
            continue
        for rel in send_matches:
            if rel in existing_remote:
                log("SKIP", f"{rel}  (already on Receive)", indent=2, out=out)
                continue
            candidates.append(rel)

    # --- logs/ dir --------------------------------------------------
    if _matches_ignore(LOGS_DIR, ignore_patterns) is not None:
        log("SKIP", f"{LOGS_DIR}/  (ignored by -i)", indent=2, out=out)
    else:
        send_logs = sorted(p for p in send_inventory if p.startswith(f"{LOGS_DIR}/"))
        if not send_logs:
            log("INFO", f"{LOGS_DIR}/  (none on Send)", indent=2, out=out)
        for rel in send_logs:
            if rel in existing_remote:
                log("SKIP", f"{rel}  (already on Receive)", indent=2, out=out)
                continue
            candidates.append(rel)

    if not candidates:
        log("OK", "receive already in sync; nothing to transfer", indent=2, out=out)
        return True, missing_globs

    # --- Batch mkdir -p on Receive (single Send -> Receive ssh) --------
    parents = sorted(
        {
            (
                f"{recv_exp_dir}/{Path(rel).parent.as_posix()}"
                if Path(rel).parent.as_posix() not in (".", "")
                else recv_exp_dir
            )
            for rel in candidates
        }
    )
    log(
        "INFO",
        f"mkdir -p Receive dirs (batch, {len(parents)} dir(s))",
        indent=2,
        out=out,
    )
    mk_args = " ".join(f"'{p}'" for p in parents)
    rc = run(
        f"{recv_auth}ssh{ssh_port_flag(recv_port)} {recv_user_host} "
        f'"mkdir -p {mk_args}"',
        dry_run=dry_run,
        out=out,
    )
    if rc != 0:
        log("FAIL", f"batch mkdir failed (rc={rc})", indent=2, out=out)
        return False, missing_globs

    # --- Direct scp per file ---------------------------------------
    log(
        "INFO",
        f"transferring {len(candidates)} file(s) Send -> Receive",
        indent=2,
        out=out,
    )
    failed: list[str] = []
    for rel in candidates:
        send_full = f"{send_exp_dir}/{rel}"
        recv_full = f"{recv_exp_dir}/{rel}"
        log("XFR", f"{rel}", indent=2, out=out)
        cmd = build_scp_cmd(
            send_path=send_full,
            recv_user_host=recv_user_host,
            recv_port=recv_port,
            recv_path=recv_full,
            recv_auth=recv_auth,
        )
        rc = run(cmd, dry_run=dry_run, out=out)
        if rc != 0:
            log("FAIL", f"{rel}  (scp rc={rc})", indent=2, out=out)
            failed.append(rel)
            continue
        log("OK", f"{rel}", indent=2, out=out)

    if failed:
        log(
            "FAIL",
            f"{len(failed)} transfer(s) failed: {failed}",
            indent=2,
            out=out,
        )
        return False, missing_globs
    log(
        "OK",
        f"transferred {len(candidates)} file(s) for {module}/{experiment}",
        indent=2,
        out=out,
    )
    return True, missing_globs


# ======================================================================
# Per-Loss_prepare direct cascade
# ======================================================================
def cascade_loss_prepare_direct(
    module: str,
    dataset: str,
    *,
    send_base: str,
    recv_user_host: str,
    recv_port: str,
    recv_base: str,
    recv_auth: str,
    dry_run: bool,
) -> bool:
    """Direct Send -> Receive push of a single ``<dataset>_Loss_prepare.json``.

    Pre-check Receive first; if already there, skip entirely. Otherwise
    ``mkdir -p`` the receive parent (Send -> Receive ssh) and ship the
    file with a single ``scp``.
    """
    fname = f"{dataset}{LOSS_PREPARE_SUFFIX}"
    send_path = f"{send_base}/{module}/{fname}"
    recv_path = f"{recv_base}/{module}/{fname}"
    recv_parent = f"{recv_base}/{module}"

    sub_header(f"LOSS_PREPARE  module={module}  dataset={dataset}")
    log("INFO", f"send    : {send_path}", indent=2)
    log("INFO", f"receive : {recv_user_host}:{recv_path}", indent=2)

    # Pre-check Send presence -- nothing to push if the source is missing.
    if not Path(send_path).is_file():
        log("MISS", f"{fname}  (not present on Send)", indent=2)
        return False

    if remote_glob_has_match(
        glob_path=recv_path,
        auth=recv_auth,
        user_host=recv_user_host,
        port=recv_port,
    ):
        log("SKIP", f"{fname}  (already on Receive)", indent=2)
        log("DONE", f"{fname} already cascaded (no-op)", indent=2)
        return True

    log("INFO", f"mkdir -p {recv_parent}", indent=2)
    rc = run(
        f"{recv_auth}ssh{ssh_port_flag(recv_port)} {recv_user_host} "
        f"\"mkdir -p '{recv_parent}'\"",
        dry_run=dry_run,
    )
    if rc != 0:
        log("FAIL", f"mkdir -p {recv_parent} (rc={rc})", indent=2)
        return False

    log("XFR", f"{fname}  Send -> Receive", indent=2)
    cmd = build_scp_cmd(
        send_path=send_path,
        recv_user_host=recv_user_host,
        recv_port=recv_port,
        recv_path=recv_path,
        recv_auth=recv_auth,
    )
    rc = run(cmd, dry_run=dry_run)
    if rc != 0:
        log("FAIL", f"{fname} scp (rc={rc})", indent=2)
        return False
    log("DONE", f"{fname} cascaded", indent=2)
    return True


# ======================================================================
# Worker: one experiment, end-to-end
# ======================================================================
def _resolve_jobs(jobs_arg: str, n: int) -> int:
    """Resolve ``-j/--jobs`` to a concrete worker count.

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


# ======================================================================
# Filesystem inventory helpers (Phase 1)
# ======================================================================
def is_relevant_file(rel: str) -> bool:
    """True iff ``rel`` is a file we actually cascade.

    The cascade transfers exactly two artifact families:
      * any path under ``logs/`` (training logs, sample dumps, etc.),
      * any path matching a glob in ``CHECKPOINT_GLOBS`` (best ckpts).

    All other files (raw step ckpts, optimiser state, scratch dumps,
    arbitrary user files) are IGNORED here -- ``has content`` and the
    diff in Phase 2 are computed over THIS set only.
    """
    if rel.startswith(f"{LOGS_DIR}/"):
        return True
    for g in CHECKPOINT_GLOBS:
        if fnmatch.fnmatchcase(rel, g):
            return True
    return False


def relevant_files(files: set[str]) -> set[str]:
    """Filter a file set down to the cascade-relevant subset."""
    return {f for f in files if is_relevant_file(f)}


def inventory_send_module(send_module_dir: Path) -> dict[str, set[str]]:
    """Walk ``<send_base>/<module>/`` LOCALLY -> {exp_name: relevant_files}.

    Each value is the set of POSIX paths (relative to the experiment
    directory) that pass ``is_relevant_file``. An experiment dir that
    exists but has no relevant files maps to an empty set -- callers
    treat that as "no content" (will be reported but not queued).
    """
    if not send_module_dir.is_dir():
        return {}
    result: dict[str, set[str]] = {}
    for exp_dir in sorted(p for p in send_module_dir.iterdir() if p.is_dir()):
        result[exp_dir.name] = relevant_files(local_list_files(exp_dir))
    return result


def inventory_recv_module(
    *,
    recv_module_dir: str,
    auth: str,
    user_host: str,
    port: str,
) -> dict[str, set[str]]:
    """Probe ``<recv_base>/<module>/`` via ONE ssh call.

    Issues a single ssh that runs TWO ``find`` calls remotely:
      1. ``find . -mindepth 1 -maxdepth 1 -type d`` -- enumerates every
         experiment subdir under <recv_module_dir>, INCLUDING empty
         ones (placeholders from previous failed/partial transfers).
      2. ``find . -mindepth 2 -type f``        -- enumerates every
         file under those subdirs.
    The two sections are separated by a sentinel line so we can parse
    both in one go without paying for two round-trips.

    Returns ``{exp_name: relevant_files_set}`` where:
      * keys = EVERY experiment subdir present on Receive (even empty
        ones -- those map to ``set()``).
      * values = subset of files that pass ``is_relevant_file`` (logs/*
        + CHECKPOINT_GLOBS matches). Irrelevant files are silently
        dropped from the value set; the key is still kept.

    Empty result + non-zero ssh exit code -> the function aborts the
    whole run with :class:`SystemExit`. Rationale: any ssh failure (auth
    denied, host unreachable, missing ``sshpass``, etc.) means we have
    NO trustworthy view of the Receive side; continuing with an empty
    inventory would let Phase 2 mis-classify every Send experiment as
    ``not on Receive`` and queue it for re-transfer (which would then
    also fail in Phase 3, or in the worst case duplicate already-pushed
    payloads). Read-only; runs unconditionally (incl. dry-run).
    """
    sentinel = "---LabToCampusServer-FILES---"
    cmd = (
        f"{auth}ssh{ssh_port_flag(port)} {user_host} "
        f"\"if [ -d '{recv_module_dir}' ]; then "
        f"cd '{recv_module_dir}' && "
        f"find . -mindepth 1 -maxdepth 1 -type d && "
        f"echo '{sentinel}' && "
        f"find . -mindepth 2 -type f; "
        f"else echo '{sentinel}'; fi\""
    )
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)

    # Fail-fast on ANY non-zero ssh exit. We do this BEFORE parsing
    # ``proc.stdout`` because a partial/garbled probe is worse than a
    # missing one: it would mix real Receive dirs with phantom ``-bash:``
    # error lines and confuse the diff. The most common causes are
    # surfaced in the message so the operator can act immediately.
    if proc.returncode != 0:
        stderr_full = (proc.stderr or "").strip()
        stderr_head = stderr_full.splitlines()[:5]
        msg_lines = [
            "",
            "=" * 78,
            "[FATAL] Receive probe failed -- aborting the cascade.",
            "=" * 78,
            f"  ssh return code : {proc.returncode}",
            f"  probed directory: {recv_module_dir!r}",
            f"  endpoint        : {user_host}{port_suffix(port)}",
            "  ssh stderr      :",
        ]
        if stderr_head:
            for line in stderr_head:
                msg_lines.append(f"      {line}")
        else:
            msg_lines.append("      <empty>")
        msg_lines.extend(
            [
                "",
                "  Common causes (check in this order):",
                "    1. Wrong $<Receiver>PWD in ./.env (Permission denied).",
                "    2. ./.env not loaded (e.g. running from a different cwd).",
                "    3. ``sshpass`` not installed on Send (rc=127, command not found).",
                "    4. Receive host unreachable / wrong $<Receiver>Host (or $<Receiver>IP).",
                "    5. Stale Receive host key (re-run with the right ~/.ssh/known_hosts).",
                "",
                "  Continuing with an empty Receive inventory would queue every Send",
                "  experiment for re-transfer, so the run is aborted now.",
                "=" * 78,
            ]
        )
        raise SystemExit("\n".join(msg_lines))

    result: dict[str, set[str]] = {}
    in_files = False
    for line in proc.stdout.splitlines():
        rel = line.strip()
        if not rel:
            continue
        if rel == sentinel:
            in_files = True
            continue
        if rel.startswith("./"):
            rel = rel[2:]
        if not in_files:
            # Section 1: experiment subdir names. Each ``rel`` is an
            # exp_name (no slash). Register the dir even if it ends up
            # empty -- the caller wants to know it exists on Receive.
            if rel and "/" not in rel:
                result.setdefault(rel, set())
            continue
        # Section 2: files beneath an exp dir. ``rel`` is
        # ``<exp_name>/<path/inside/exp>``.
        parts = rel.split("/", 1)
        if len(parts) < 2:
            continue
        exp_name, rel_in_exp = parts
        result.setdefault(exp_name, set())
        if is_relevant_file(rel_in_exp):
            result[exp_name].add(rel_in_exp)

    return result


# ======================================================================
# Worker: transfer ONE experiment, write JSON log
# ======================================================================
def _transfer_one_to_json(
    method: str,
    module: str,
    experiment: str,
    *,
    send_base: str,
    recv_user_host: str,
    recv_port: str,
    recv_base: str,
    recv_auth: str,
    # FULL Send file set (not pre-filtered) -- transfer_experiment does
    # its own glob/log filtering and ignore-pattern logic on top of it.
    send_inventory: set[str],
    # Receive relevant-file set (subset already on the remote side); used
    # to compute the delta to push during this run.
    existing_remote: set[str],
    ignore_patterns: list[str],
    dry_run: bool,
    log_dir: Path,
) -> dict:
    """Run :func:`transfer_experiment` for ONE exp into a JSON record.

    All terminal-style log lines are captured into a ``StringIO`` and
    embedded in the JSON file written at
    ``<log_dir>/<method>__<module>__<experiment>.json``.
    The TERMINAL never sees the per-file scp / mkdir output.

    Returns a concise summary dict containing at minimum:
      * ``method``, ``module``, ``experiment``
      * ``send_path``, ``recv_path``
      * ``started_at`` / ``finished_at`` / ``duration_seconds``
      * ``send_files_total`` / ``send_files_relevant`` / ``recv_files_existing``
      * ``files_to_transfer`` (count after diff)
      * ``missing_globs``
      * ``result`` -- one of ``OK`` / ``FAIL`` / ``ERROR``
      * ``json_path`` -- absolute path to the written JSON file
      * ``log`` (only in the JSON file, NOT echoed to terminal)
    """
    started = datetime.now()
    started_mono = time.monotonic()
    buf = io.StringIO()

    send_relev = relevant_files(send_inventory)
    pending = sorted(send_relev - existing_remote)

    record: dict = {
        "method": method,
        "module": module,
        "experiment": experiment,
        "send_path": f"{send_base}/{module}/{experiment}",
        "recv_path": f"{recv_base}/{module}/{experiment}",
        "started_at": started.isoformat(timespec="seconds"),
        "send_files_total": len(send_inventory),
        "send_files_relevant": len(send_relev),
        "recv_files_existing": len(existing_remote),
        "files_to_transfer": len(pending),
        "pending_files": pending,
        "dry_run": dry_run,
    }

    error: str | None = None
    ok = False
    missing_globs: list[str] = []
    try:
        ok, missing_globs = transfer_experiment(
            module,
            experiment,
            send_base=send_base,
            recv_user_host=recv_user_host,
            recv_port=recv_port,
            recv_base=recv_base,
            recv_auth=recv_auth,
            existing_remote=existing_remote,
            send_inventory=send_inventory,
            ignore_patterns=ignore_patterns,
            dry_run=dry_run,
            out=buf,
        )
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
        ok = False

    finished = datetime.now()
    record["finished_at"] = finished.isoformat(timespec="seconds")
    record["duration_seconds"] = round(time.monotonic() - started_mono, 3)
    record["missing_globs"] = missing_globs
    if error is not None:
        record["result"] = "ERROR"
        record["error"] = error
    else:
        record["result"] = "OK" if ok else "FAIL"
    record["log"] = buf.getvalue().splitlines()

    log_dir.mkdir(parents=True, exist_ok=True)
    json_path = log_dir / f"{method}__{module}__{experiment}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    record["json_path"] = str(json_path)
    return record


# ======================================================================
# Main
# ======================================================================
def main() -> int:
    """Entry point: inventory, diff, parallel transfer to receiver."""
    args = parse_args()

    if args.module is None:
        # No -m given -- fall back to the configured defaults. This is
        # the ONLY implicit form; once -m is set, parse_module_spec
        # requires the full <method>/<module>.
        method = DEFAULT_METHOD
        module_choice = ALL_KEYWORD
    else:
        method, module_choice = parse_module_spec(args.module)
    configs_root = configs_root_for(method)

    # ---- Resolve endpoints from --sender/--receiver profiles ------
    # Each field has a CLI inline override; when omitted, it is filled
    # from the matching profile in ``./.env``. ``--sender`` only
    # supplies ``--send-storage-root``; ``--receiver`` supplies SSH +
    # password + ``--receive-storage-root``.
    send_profile: dict[str, str] = (
        load_server_profile(args.sender) if args.sender else {}
    )
    recv_profile: dict[str, str] = (
        load_server_profile(args.receiver) if args.receiver else {}
    )
    if not args.send_storage_root and send_profile:
        args.send_storage_root = send_profile["StorageRoot"]
    if not args.receive_ssh and recv_profile:
        args.receive_ssh = profile_ssh(recv_profile)
    if not args.receive_storage_root and recv_profile:
        args.receive_storage_root = recv_profile["StorageRoot"]
    if args.password is None:
        args.password = (recv_profile["PWD"] if recv_profile else None) or None

    # ---- Validate Receive identity (env-driven defaults) -----------
    # ``--receive-ssh`` and the storage roots are required, but they
    # default to values pulled from the named profiles so the normal
    # invocation needs no CLI flags. If both are empty, fail fast with
    # a single, clear message that points at both fix paths (edit
    # ``./.env`` profile OR pass the flag).
    if not args.receive_ssh:
        raise SystemExit(
            "[ERROR] Receive SSH endpoint is unset. Pass --receiver "
            "<Profile> (with ``<P>User`` and ``<P>Host``/``<P>IP`` set "
            "in ./.env), OR pass "
            "``--receive-ssh 'ssh -p <port> <user>@<host>'``."
        )
    if not args.send_storage_root:
        raise SystemExit(
            "[ERROR] Send storage root is unset. Pass --sender <Profile> "
            f"(with ``{args.sender or '<P>'}StorageRoot`` set in ./.env), "
            "OR pass "
            "``--send-storage-root <abs path>``. There is no hardcoded fallback."
        )
    if not args.receive_storage_root:
        raise SystemExit(
            "[ERROR] Receive storage root is unset. Pass --receiver "
            f"<Profile> (with ``{args.receiver or '<P>'}StorageRoot`` set "
            "in ./.env), OR pass "
            "``--receive-storage-root <abs path>``. There is no hardcoded fallback."
        )

    recv_port, recv_user_host = parse_ssh(args.receive_ssh)

    # Storage roots are taken AS-IS; ``EXPERIMENT/`` is NOT prepended.
    # We append ONLY the per-method subdir.
    send_storage_root = args.send_storage_root.rstrip("/")
    recv_storage_root = args.receive_storage_root.rstrip("/")
    send_base = f"{send_storage_root}/{method}"
    recv_base = f"{recv_storage_root}/{method}"

    recv_auth = build_auth_prefix(args.password)
    auth_mode = "password (sshpass)" if args.password else "plain ssh/scp"

    if module_choice == ALL_KEYWORD:
        modules = list(VALID_MODULES)
    else:
        modules = [module_choice]

    if args.dataset is None:
        ds_arg: list[str] = [ALL_KEYWORD]
    else:
        ds_arg = list(args.dataset)

    # ---- JSON log directory (flat layout; re-runs overwrite) -------
    log_root = Path(send_storage_root) / LOG_OUTPUT_SUBDIR

    header(
        "LabToCampusServer  --  SendServer (LOCAL) -scp-> ReceiveServer (any two hosts via .env)"
    )
    log("INFO", f"sender profile     : {args.sender or '<inline overrides>'}")
    log("INFO", f"receiver profile   : {args.receiver or '<inline overrides>'}")
    log("INFO", "SendServer    (src) : LOCAL filesystem (this host)")
    log("INFO", f"  send_storage_root : {send_storage_root}")
    log("INFO", f"  send_base         : {send_base}")
    log("INFO", f"ReceiveServer (dst) : {recv_user_host}{port_suffix(recv_port)}")
    log("INFO", f"  recv_storage   : {recv_storage_root}")
    log("INFO", f"  recv_base      : {recv_base}")
    log("INFO", f"method             : {method}")
    log("INFO", f"  configs_root     : {configs_root}")
    log("INFO", f"modules            : {modules}")
    log("INFO", f"experiment         : {args.experiment}")
    log("INFO", f"datasets           : {ds_arg}")
    log("INFO", f"ignore             : {args.ignore}")
    log("INFO", f"auth (receive)      : {auth_mode}")
    log("INFO", f"jobs               : {args.jobs}")
    log("INFO", f"dry_run            : {args.dry_run}")
    log("INFO", f"json_log_dir       : {log_root}")
    hr()

    # ==================================================================
    # PHASE 1: INVENTORY (both sides, by module)
    # ==================================================================
    print()
    header("PHASE 1  INVENTORY  (both sides, by module)")
    log("INFO", "sources scanned in this phase:")
    log("INFO", f"  YAML configs (Send, local) : {configs_root.resolve()}")
    log("INFO", f"  Send filesystem (LOCAL)    : {Path(send_base).resolve()}")
    log(
        "INFO",
        f"  Receive filesystem (remote): "
        f"{recv_user_host}:{recv_base}{port_suffix(recv_port)}",
    )
    print()

    # YAML candidates -- restrict the consideration set to experiments
    # that the local repo actually declares (avoids shipping stale dirs).
    log(
        "INFO",
        f"scanning YAML configs under: {configs_root.resolve()}",
    )
    yaml_candidates: dict[str, set[str]] = {}
    for module in modules:
        names = discover_experiments(module, configs_root)
        yaml_candidates[module] = set(names)
        log(
            "INFO",
            f"  {method}/{module:9s}: {len(names):3d} candidate(s)  "
            f"[from {configs_root.as_posix()}/**/train_{module}_*.yml]",
        )
    if args.experiment != ALL_KEYWORD:
        log(
            "INFO",
            f"-e set: restricting consideration to '{args.experiment}'",
        )

    # ----- Send side (local filesystem) -----
    print()
    log(
        "INFO",
        f"scanning SendServer (local filesystem) under: {send_base}",
    )
    send_inventory: dict[str, dict[str, set[str]]] = {}
    for module in modules:
        send_module_dir = Path(send_base) / module
        inv = inventory_send_module(send_module_dir)
        # Restrict to YAML candidates (when -e all) or to single -e.
        if args.experiment == ALL_KEYWORD:
            cand = yaml_candidates[module]
            inv = {k: v for k, v in inv.items() if k in cand}
        else:
            inv = {k: v for k, v in inv.items() if k == args.experiment}
        send_inventory[module] = inv
        with_content = sum(1 for v in inv.values() if v)
        log(
            "INFO",
            f"  {method}/{module:9s}: {len(inv):3d} dir(s) on Send, "
            f"{with_content:3d} with content  [scanned: {send_module_dir}]",
        )

    # ----- Receive side (one ssh per module) -----
    print()
    log(
        "INFO",
        f"probing ReceiveServer (one ssh per module) under: "
        f"{recv_user_host}:{recv_base}{port_suffix(recv_port)}",
    )
    recv_inventory: dict[str, dict[str, set[str]]] = {}
    for module in modules:
        recv_module_dir = f"{recv_base}/{module}"
        inv = inventory_recv_module(
            recv_module_dir=recv_module_dir,
            auth=recv_auth,
            user_host=recv_user_host,
            port=recv_port,
        )
        # IMPORTANT: do NOT filter Receive inventory by yaml_candidates.
        # Phase 2 only iterates Send keys; Receive extras are harmless
        # lookups, but FILTERING here would silently drop real Receive
        # dirs whose YAML configs no longer exist locally, leading to
        # spurious re-transfers. ``-e <name>`` still scopes the probe.
        if args.experiment != ALL_KEYWORD:
            inv = {k: v for k, v in inv.items() if k == args.experiment}
        recv_inventory[module] = inv
        with_content = sum(1 for v in inv.values() if v)
        empty_dirs = len(inv) - with_content
        log(
            "INFO",
            f"  {method}/{module:9s}: {len(inv):3d} dir(s) on Receive, "
            f"{with_content:3d} with content, {empty_dirs:3d} empty  "
            f"[probed: {recv_user_host}:{recv_module_dir}]",
        )

    # ==================================================================
    # PHASE 2: DIFF (skip vs queue)
    # ==================================================================
    print()
    header("PHASE 2  DIFF  (skip already-on-Receive, queue the rest)")
    log("INFO", "comparing Send-relevant vs Receive-relevant per experiment:")
    log("INFO", f"  Send base    : {send_base}")
    log("INFO", f"  Receive base : {recv_user_host}:{recv_base}")
    print()

    # Each queued item carries the FULL Send inventory + Receive relevant
    # set so the Phase-3 worker can re-use them without re-walking.
    skipped: list[tuple[str, str, str]] = []
    queued: list[dict] = []
    no_lab: list[tuple[str, str, str]] = []

    for module in modules:
        labs = send_inventory[module]
        camps = recv_inventory[module]
        for exp in sorted(labs):
            send_rel = labs[exp]
            if not send_rel:
                no_lab.append((method, module, exp))
                continue
            camp_rel = camps.get(exp, set())
            missing = send_rel - camp_rel
            label = f"{method}/{module}/{exp}"
            if not missing:
                log(
                    "SKIP",
                    f"{label}  (Receive already has all {len(send_rel)} file(s))",
                )
                skipped.append((method, module, exp))
            else:
                log(
                    "XFR ",
                    f"{label}  ({len(missing)} of {len(send_rel)} file(s) missing)",
                )
                # Re-walk the FULL Send dir (not just relevant) -- the
                # Phase-3 worker re-uses transfer_experiment which does
                # its own glob/log filtering and ignore-pattern logic.
                send_full = local_list_files(Path(f"{send_base}/{module}/{exp}"))
                queued.append(
                    {
                        "method": method,
                        "module": module,
                        "experiment": exp,
                        "send_inventory": send_full,
                        "existing_remote": camp_rel,
                        "n_missing": len(missing),
                    }
                )

    if no_lab:
        print()
        log("INFO", f"Send dirs with NO relevant content: {len(no_lab)}")
        for _m, mod, exp in no_lab:
            log("INFO", f"  - {method}/{mod}/{exp}  (empty / no logs+ckpts)", indent=0)

    print()
    log("INFO", f"already on Receive (skip)  : {len(skipped)}")
    log("INFO", f"need to send (queue)      : {len(queued)}")

    # ==================================================================
    # PHASE 3: PARALLEL TRANSFER (JSON logs)
    # ==================================================================
    overall_failed: list[str] = []
    overall_results: list[dict] = []

    if not queued:
        print()
        log("INFO", "nothing to transfer in Phase 3 (Receive is up to date)")
    else:
        jobs = _resolve_jobs(args.jobs, len(queued))
        print()
        header(f"PHASE 3  PARALLEL TRANSFER  ({len(queued)} exp, workers={jobs})")
        log("INFO", f"source (Send)   : {send_base}")
        log(
            "INFO",
            f"target (Receive): {recv_user_host}:{recv_base}"
            f"{port_suffix(recv_port)}",
        )
        log("INFO", f"json_log_dir   : {log_root}")
        log("INFO", f"workers        : {jobs}")
        if args.dry_run:
            log(
                "INFO",
                "DRY-RUN: scp/mkdir commands will be printed into the JSON log only.",
            )
        print()

        run_started = time.monotonic()

        def _do_one(item: dict) -> dict:
            return _transfer_one_to_json(
                item["method"],
                item["module"],
                item["experiment"],
                send_base=send_base,
                recv_user_host=recv_user_host,
                recv_port=recv_port,
                recv_base=recv_base,
                recv_auth=recv_auth,
                send_inventory=item["send_inventory"],
                existing_remote=item["existing_remote"],
                ignore_patterns=args.ignore,
                dry_run=args.dry_run,
                log_dir=log_root,
            )

        total = len(queued)
        if jobs <= 1 or total <= 1:
            for i, item in enumerate(queued, start=1):
                label = f"{item['method']}/{item['module']}/{item['experiment']}"
                log(
                    "XFR ",
                    f"[{i}/{total}] {label} starting ({item['n_missing']} file(s))...",
                )
                rec = _do_one(item)
                tag = "OK" if rec["result"] == "OK" else "FAIL"
                log(
                    tag,
                    f"[{i}/{total}] {label}  {rec['duration_seconds']:.1f}s  "
                    f"-> {rec['json_path']}",
                )
                overall_results.append(rec)
                if rec["result"] != "OK":
                    overall_failed.append(label)
        else:
            done = 0
            with ThreadPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(_do_one, it): it for it in queued}
                for fut in as_completed(futures):
                    item = futures[fut]
                    label = f"{item['method']}/{item['module']}/{item['experiment']}"
                    done += 1
                    try:
                        rec = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        with _OUTPUT_LOCK:
                            log(
                                "FAIL",
                                f"[{done}/{total}] {label}  worker raised: {exc!r}",
                            )
                            sys.stdout.flush()
                        overall_failed.append(label)
                        continue
                    tag = "OK" if rec["result"] == "OK" else "FAIL"
                    with _OUTPUT_LOCK:
                        log(
                            tag,
                            f"[{done}/{total}] {label}  {rec['duration_seconds']:.1f}s  "
                            f"-> {rec['json_path']}",
                        )
                        sys.stdout.flush()
                    overall_results.append(rec)
                    if rec["result"] != "OK":
                        overall_failed.append(label)

        run_dt = time.monotonic() - run_started
        print()
        log("INFO", f"Phase 3 wall time : {run_dt:.1f}s")

    # ==================================================================
    # PHASE 4: LOSS_PREPARE PASS (per-dataset, sequential, terminal log)
    # ==================================================================
    overall_partial: list[str] = []
    for module in modules:
        effective_ds = [d for d in ds_arg if d != ""]
        if not effective_ds:
            print()
            log(
                "SKIP",
                f"{method}/{module}: Loss_prepare pass skipped (-d '')",
            )
            continue
        if ALL_KEYWORD in effective_ds:
            datasets = discover_datasets(module, configs_root)
            if not datasets:
                print()
                log(
                    "WARN",
                    f"no datasets discovered for {method}/{module} "
                    f"under {configs_root}",
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
            f"{method}/{module}: Loss_prepare pass over "
            f"{len(datasets)} dataset(s): {datasets}",
        )
        for ds in datasets:
            ok = cascade_loss_prepare_direct(
                module,
                ds,
                send_base=send_base,
                recv_user_host=recv_user_host,
                recv_port=recv_port,
                recv_base=recv_base,
                recv_auth=recv_auth,
                dry_run=args.dry_run,
            )
            if not ok:
                overall_failed.append(f"{module}/{ds}{LOSS_PREPARE_SUFFIX}")

    # Collect partial-glob warnings from Phase-3 records.
    for rec in overall_results:
        if rec["missing_globs"]:
            overall_partial.append(
                f"{rec['method']}/{rec['module']}/{rec['experiment']}  "
                f"missing={rec['missing_globs']}"
            )

    # ==================================================================
    # PHASE 5: SUMMARY
    # ==================================================================
    print()
    header("SUMMARY")
    n_scanned = sum(len(send_inventory[m]) for m in modules)
    n_ok = sum(1 for r in overall_results if r["result"] == "OK")
    log("INFO", f"experiments scanned   : {n_scanned}")
    log("INFO", f"empty Send dirs (skip) : {len(no_lab)}")
    log("INFO", f"already on Receive     : {len(skipped)}")
    log("INFO", f"queued for transfer   : {len(queued)}")
    log("INFO", f"  succeeded           : {n_ok}")
    log("INFO", f"  failed              : {len(queued) - n_ok}")
    log("INFO", f"partial (send miss)    : {len(overall_partial)}")
    for x in overall_partial:
        log("WARN", f"  - {x}")
    log("INFO", f"failed transfers      : {len(overall_failed)}")
    for x in overall_failed:
        log("FAIL", f"  - {x}")
    if queued:
        log("INFO", f"json log dir          : {log_root}")
    if not overall_failed:
        if args.dry_run:
            log("DONE", "dry-run complete (no real transfers performed)")
        elif queued:
            log("DONE", "all queued items cascaded successfully")
        else:
            log("DONE", "nothing to do -- Receive is already up to date")
    hr()

    return 0 if not overall_failed else 1


if __name__ == "__main__":
    sys.exit(main())
