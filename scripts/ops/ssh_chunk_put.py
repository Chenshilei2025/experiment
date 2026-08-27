"""Copy a file over SSH by streaming bytes to a remote Python receiver.

This is a fallback for environments where scp/rsync hang behind VPN or NAT.
It verifies only byte count and JSONL line count, not hashes.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import textwrap


def _ssh_command(args: argparse.Namespace) -> list[str]:
    receiver = textwrap.dedent(
        """
        import os
        import pathlib
        import sys

        dest = pathlib.Path(sys.argv[1])
        tmp = dest.with_name(dest.name + ".tmp_chunk_put")
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = sys.stdin.buffer.read()
        tmp.write_bytes(data)
        os.replace(tmp, dest)
        size = dest.stat().st_size
        if dest.suffix == ".jsonl":
            lines = sum(1 for line in dest.read_text(encoding="utf-8").splitlines() if line.strip())
            print(f"REMOTE_FILE_READY path={dest} bytes={size} nonempty_lines={lines}")
        else:
            print(f"REMOTE_FILE_READY path={dest} bytes={size}")
        """
    ).strip()
    command = [
        "ssh",
        "-o",
        args.ssh_option,
        "-p",
        str(args.port),
        args.host,
        "python3",
        "-c",
        receiver,
        args.remote_path,
    ]
    if args.identity:
        command[1:1] = ["-i", str(args.identity)]
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="SSH target, for example root@10.220.5.101")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ssh-option", default="KexAlgorithms=curve25519-sha256")
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--remote-path", required=True)
    args = parser.parse_args()

    if not args.local.is_file():
        parser.error(f"local file does not exist: {args.local}")

    with args.local.open("rb") as handle:
        result = subprocess.run(_ssh_command(args), stdin=handle, capture_output=True, text=False)
    if result.stdout:
        print(result.stdout.decode("utf-8", errors="replace"), end="")
    if result.stderr:
        print(result.stderr.decode("utf-8", errors="replace"), end="", file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
