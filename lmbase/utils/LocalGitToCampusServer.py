#!/usr/bin/env python3
"""Ship a GitHub repo (clone-and-zip) to ANY ReceiveServer.

Arguments (every flag must be explicit on the CLI -- no hidden defaults):
    --receiver PROFILE        REQUIRED. .env profile for the DESTINATION host.
    -s / --ssh SSH_STR        Override receiver SSH string from .env.
    -p / --password PWD       Receive SSH password override (profile PWD if omitted).
    -d / --remote-dir PATH    Override receiver CodeRoot from .env.
    -r / --repo NAME          Repository name (default: current directory name).
    --prefix URL              GitHub org prefix for building clone URL.

Effects:
    Receiver : resolved from ./.env[<receiver>{IP,User,Host,Port,PWD,CodeRoot}]
    Pipeline : clone -> zip -> scp -> unzip on ReceiveServer
    Idempotent : zip cached locally; re-run skips clone+zip if present

Usage::

    python3 LocalGitToCampusServer.py --receiver CampusServer -p '...'
    python3 LocalGitToCampusServer.py --receiver CampusServer -r Reasoning-Autoregressive-Modeling
    python3 LocalGitToCampusServer.py --receiver LabServer -d /home/user/code

----------------------------------------------------------------------
Detailed description
----------------------------------------------------------------------
This script DEPLOYS THE CODE itself (a fresh git clone of an upstream
GitHub repo, archived as a zip) onto a remote ReceiveServer. It is the
COMPANION to ``LabToCampusServer.py`` / ``LabToLocalToCompusServer.py``,
which ship TRAINING ARTIFACTS. Different concern, different
destination dir, no overlap with the experiment cascade -- this script
does NOT read or depend on ``configs/`` and is unrelated to the
``ModelLearn`` template.

Pipeline:
    1. Fresh-clone ``<prefix><repo>.git`` into ``./.tmpl/``.
    2. Zip the clone into ``./.tmpl/<repo>.zip`` (excluding ``.git`` / ``__MACOSX``).
    3. ``ssh ... mkdir -p <remote_dir>`` so the destination tree is auto-created
       if it does not already exist (idempotent; safe to re-run).
    4. ``scp`` the zip to the ReceiveServer.
    5. ``ssh`` in, ``unzip -o``, strip ``__MACOSX`` dirs and the zip itself.
    6. Remove the local ``./.tmpl/`` only on success.  On failure the zip is
       kept so a retry skips the clone+zip steps automatically.

----------------------------------------------------------------------
PRE-FLIGHT SETUP  (do this ONCE per workstation; deploy is .env-driven)
----------------------------------------------------------------------
The destination is selected by NAMED SERVER PROFILE (``--receiver
CampusServer`` etc.) -- there is NO hardcoded host / path / port in
this file. Append a new profile to ``./.env`` and pass its name on the
CLI; no source-code edit needed. Missing values fail fast at startup
(no silent fallback).

Declare one or more SERVER PROFILES in ``./.env``. Each profile is a
flat namespace ``<Profile><Attr>``; this script consumes ``IP``,
``User``, ``Host``, ``Port``, ``PWD`` and ``CodeRoot``. Example::

    # Profile: CampusServer  (HPC cluster at HKUST-GZ)
    CampusServerIP       = 10.120.48.27
    CampusServerUser     = sijiachen
    CampusServerHost     = hpc3login.hpc.hkust-gz.edu.cn
    CampusServerPort     =                          # OPTIONAL
    CampusServerPWD      = <password>               # OPTIONAL
    CampusServerCodeRoot = /home/sijiachen/code/projects

Add as many profiles as you have destinations. ``Host`` falls back
to ``IP`` if not separately set.

From the resolved profile this script derives at startup (visible in the banner)::

    ssh         = ``ssh [-p <Port>] <User>@<Host>``
    remote_dir  = ``$<Profile>CodeRoot``
    auth        = password (sshpass) when ``$<Profile>PWD`` is set,
                  else plain ssh/scp (no -p)

Any CLI flag (``-s`` / ``-p`` / ``-d``) takes precedence over the profile
values.  ``--receiver`` is REQUIRED -- there is no default.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from dotenv import load_dotenv

# --- .env loading ------------------------------------------------------
# Load credentials / profile attributes from ``./.env`` at the repo
# root BEFORE any default is computed below. We never hard-code
# username, hostname or password in this source file -- everything
# sensitive flows through ``./.env``. ``override=False`` so a real env
# var (exported in the shell) wins over the file, matching typical CI
# conventions.
SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=SCRIPT_DIR / ".env", override=False)

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
            "[FATAL] empty server profile name; pass --receiver <Profile>. "
            "There is no default -- the flag is REQUIRED."
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
    env key and the matching ``-s`` CLI override.
    """
    user, host, port = profile["User"], profile["Host"], profile["Port"]
    if not user or not host:
        return ""
    return f"ssh -p {port} {user}@{host}" if port else f"ssh {user}@{host}"


DEFAULT_PREFIX = "git@github.com:AgenticFinLab/"
DEFAULT_REPO = "Reasoning-Autoregressive-Modeling"
TMPL_DIR_NAME = ".tmpl"


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
            ``--ssh`` flag or env value).
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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for git-to-server deployment."""
    parser = argparse.ArgumentParser(
        description=(
            "Clone a GitHub repo and ship a zip to ANY ReceiveServer. "
            "Pick the destination by named profile via ``--receiver "
            "<Profile>`` (resolved from ``./.env``)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--receiver",
        default="",
        metavar="PROFILE",
        help=(
            "Named server profile (from ``./.env``) to deploy INTO. "
            "Resolves SSH endpoint + password + remote dir from "
            "``<PROFILE>{IP,User,Host,Port,PWD,CodeRoot}``. REQUIRED."
        ),
    )
    parser.add_argument(
        "-s",
        "--ssh",
        default="",
        help=(
            "SSH connection string of the form 'ssh [-p PORT] USER@HOST'. "
            "Inline override; when empty, built from the --receiver profile."
        ),
    )
    parser.add_argument(
        "-p",
        "--password",
        default=None,
        help=(
            "Receiver SSH login password. Inline override; when omitted, "
            "read from the --receiver profile's ``PWD`` attribute. Pass "
            "``-p ''`` (empty) to force key-based auth and ignore the "
            "profile value -- any failure then surfaces at runtime (no "
            "preemptive validation)."
        ),
    )
    parser.add_argument(
        "-d",
        "--remote-dir",
        default="",
        help=(
            "Remote target directory. Inline override; when empty, read "
            "from the --receiver profile's ``CodeRoot`` attribute."
        ),
    )
    parser.add_argument(
        "-r",
        "--repo",
        default=DEFAULT_REPO,
        help=f"Repository name (no prefix, no .git). Default: '{DEFAULT_REPO}'.",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help=f"GitHub org prefix used to build clone URL. Default: '{DEFAULT_PREFIX}'.",
    )
    return parser.parse_args()


def run(cmd: str) -> None:
    """Run a shell command, streaming output; raise on non-zero exit."""
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def ensure_zip_available() -> None:
    """Fail fast if the local ``zip`` binary is missing."""
    if shutil.which("zip") is None:
        raise SystemExit(
            "[ERROR] 'zip' command not found. Install it (macOS: ships by default; "
            "Debian/Ubuntu: apt-get install zip) and retry."
        )


def resolve_receiver(args: argparse.Namespace) -> tuple[str, str | None, str]:
    """Resolve ``(ssh_string, password, remote_dir)`` from CLI + profile.

    Each field has a CLI inline override; when omitted, it is filled
    from the ``--receiver`` profile in ``./.env``. All three must be
    non-empty (well-defined) by the time we return; otherwise a single
    fail-fast message names every missing piece and both fixes.
    """
    profile: dict[str, str] = {}
    if args.receiver:
        profile = load_server_profile(args.receiver)

    if args.ssh:
        ssh = args.ssh
    elif profile:
        ssh = profile_ssh(profile)
    else:
        ssh = ""

    if args.password is not None:
        password = args.password
    elif profile and profile["PWD"]:
        password = profile["PWD"]
    else:
        password = None

    remote_dir = args.remote_dir or (profile["CodeRoot"] if profile else "")

    if not ssh:
        raise SystemExit(
            "[FATAL] receiver SSH string is empty. Pass --receiver "
            "<Profile> (with ``<P>User`` and ``<P>Host``/``<P>IP`` set in "
            "./.env), or pass "
            "-s 'ssh [-p PORT] USER@HOST'."
        )
    if not remote_dir:
        raise SystemExit(
            "[FATAL] remote directory is empty. Set "
            f"``{args.receiver or '<Profile>'}CodeRoot`` in ./.env, or "
            "pass -d /abs/path. There is no hardcoded fallback."
        )
    return ssh, password, remote_dir


def main() -> int:
    """Entry point: clone, zip, and deploy to receiver server."""
    args = parse_args()
    ssh_str, password, remote_dir = resolve_receiver(args)

    port, user_host = parse_ssh(ssh_str)
    repo = args.repo
    prefix = args.prefix

    repo_url = f"{prefix}{repo}.git"
    tmpl_dir = SCRIPT_DIR / TMPL_DIR_NAME
    clone_path = tmpl_dir / repo
    zip_name = f"{repo}.zip"
    zip_path = tmpl_dir / zip_name

    auth_mode = "password (sshpass)" if password else "plain ssh/scp (no -p)"
    print("=" * 70)
    print(f"[CONFIG] receiver   = {args.receiver or '<inline overrides>'}")
    print(f"[CONFIG] repo       = {repo}")
    print(f"[CONFIG] repo_url   = {repo_url}")
    print(f"[CONFIG] tmpl_dir   = {tmpl_dir}")
    print(
        f"[CONFIG] ssh        = ssh{ssh_port_flag(port)} {user_host}{port_suffix(port)}"
    )
    print(f"[CONFIG] remote_dir = {remote_dir}")
    print(f"[CONFIG] auth       = {auth_mode}")
    print("=" * 70)

    ensure_zip_available()
    auth = build_auth_prefix(password)

    # Reuse existing .tmpl/ zip from a previous run if present AND valid.
    if zip_path.exists():
        # Validate zip integrity before reuse (corrupt/truncated zip from
        # interrupted run would silently deploy broken content to remote).
        if not zipfile.is_zipfile(zip_path):
            print(
                f"[WARN] {zip_path} exists but is NOT a valid zip "
                f"(corrupt/truncated?) -- rebuilding."
            )
            shutil.rmtree(tmpl_dir)
            tmpl_dir.mkdir(parents=True)
        else:
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    bad = zf.testzip()
                if bad is not None:
                    print(
                        f"[WARN] {zip_path} has corrupt entry: {bad} " f"-- rebuilding."
                    )
                    shutil.rmtree(tmpl_dir)
                    tmpl_dir.mkdir(parents=True)
                else:
                    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
                    print(
                        f"[REUSE] Verified zip: {zip_path} "
                        f"({zip_size_mb:.1f} MB) -- skipping clone+zip."
                    )
            except (zipfile.BadZipFile, OSError) as exc:
                print(
                    f"[WARN] {zip_path} failed integrity check: {exc} "
                    f"-- rebuilding."
                )
                shutil.rmtree(tmpl_dir)
                tmpl_dir.mkdir(parents=True)

    if not zip_path.exists():
        # Fresh .tmpl/
        if tmpl_dir.exists():
            shutil.rmtree(tmpl_dir)
        tmpl_dir.mkdir(parents=True)

        # 1. Clone (shallow to keep the zip small).
        print(f"\n[1/5] Cloning {repo_url} into {clone_path}")
        run(f"git clone --depth 1 {repo_url} {clone_path}")

        # 2. Zip the clone (exclude .git and __MACOSX).
        print(f"\n[2/5] Zipping into {zip_path}")
        run(
            f"cd {tmpl_dir} && zip -r {zip_name} {repo} "
            f"-x '{repo}/.git/*' '*/__MACOSX/*'"
        )

    try:

        # 3. Ensure the remote dir exists (auto-create, idempotent).
        print(f"\n[3/5] ssh mkdir -p {remote_dir}")
        run(f"{auth}ssh{ssh_port_flag(port)} {user_host} \"mkdir -p '{remote_dir}'\"")

        # 4. scp the zip.
        print(f"\n[4/5] scp {zip_path} -> {user_host}:{remote_dir}/")
        run(f"{auth}scp{scp_port_flag(port)} {zip_path} {user_host}:{remote_dir}/")

        # 5. Unzip + cleanup on the remote.
        print("\n[5/5] Unzip + cleanup on remote")
        remote_cmd = (
            f"cd {remote_dir} && unzip -o '*.zip' "
            f"&& find . -type d -name '__MACOSX' -exec rm -rf {{}} + "
            f"&& rm -f *.zip "
            "&& echo 'Done: extracted, cleaned __MACOSX and removed .zip'"
        )
        run(f'{auth}ssh{ssh_port_flag(port)} {user_host} "{remote_cmd}"')

        # Only remove .tmpl/ on success — keep it on failure so the
        # zip can be re-submitted without repeating the clone+zip steps.
        if tmpl_dir.exists():
            print(f"\n[cleanup] Removing local {tmpl_dir}")
            shutil.rmtree(tmpl_dir)

        print("\n[OK] Deployment finished.")
        return 0
    except Exception:
        print(
            f"\n[FAILED] .tmpl/ kept at {tmpl_dir} for retry.\n"
            f"         Re-run the script and it will reuse the existing zip."
        )
        raise


if __name__ == "__main__":
    sys.exit(main())
