"""Small, dependency-free provenance and console logging for experiments."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
from typing import Iterator, Mapping, TextIO


_SECRET_MARKERS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTHORIZATION", "COOKIE")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_run_provenance(
    output_dir: Path, *, command: list[str] | None = None, environment: Mapping[str, str] | None = None,
) -> None:
    """Write reproducibility metadata without persisting service credentials."""
    _write_json(output_dir / "command.json", {
        "argv": list(sys.argv) if command is None else command,
        "cwd": str(Path.cwd()),
        "python_executable": sys.executable,
    })
    environment = {
        name: ("<redacted>" if any(marker in name.upper() for marker in _SECRET_MARKERS) else value)
        for name, value in sorted((os.environ if environment is None else environment).items()) if name.startswith("LOYAL_")
    }
    _write_json(output_dir / "environment.json", environment)


class _Tee:
    def __init__(self, original: TextIO, log: TextIO) -> None:
        self._original, self._log = original, log

    def write(self, data: str) -> int:
        self._original.write(data)
        self._log.write(data)
        return len(data)

    def flush(self) -> None:
        self._original.flush()
        self._log.flush()

    def isatty(self) -> bool:
        return self._original.isatty()


@contextmanager
def capture_run_output(output_dir: Path) -> Iterator[Path]:
    """Mirror Python stdout/stderr to a run-local log for the current process."""
    path = output_dir / "run.log"
    with path.open("a", encoding="utf-8", buffering=1) as log:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(stdout, log), _Tee(stderr, log)  # type: ignore[assignment]
        try:
            yield path
        finally:
            sys.stdout, sys.stderr = stdout, stderr
