"""Bootstrap creative SFT data from Hugging Face parquet sources."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from scripts.data.prepare_creative_slime import build_creative_file


DEFAULT_WRITINGPROMPTS_REPO = "euclaise/writingprompts"
DEFAULT_ROCSTORIES_REPO = "hamishivi/ROCStories"


def _download_dataset(repo_id: str, *, allow_patterns: list[str], target_root: Path) -> list[Path]:
    snapshot_root = Path(
        snapshot_download(
            repo_id,
            repo_type="dataset",
            allow_patterns=allow_patterns,
            local_dir=target_root / repo_id.replace("/", "__"),
            local_dir_use_symlinks=False,
        )
    )
    return sorted(snapshot_root.glob("data/**/*.parquet"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("artifacts/cache/creative_sources"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--writingprompts-repo", default=DEFAULT_WRITINGPROMPTS_REPO)
    parser.add_argument("--rocstories-repo", default=DEFAULT_ROCSTORIES_REPO)
    parser.add_argument("--writingprompts-limit", type=int, default=512)
    parser.add_argument("--rocstories-limit", type=int, default=512)
    args = parser.parse_args()

    cache_root = args.cache_root
    cache_root.mkdir(parents=True, exist_ok=True)
    writingprompts_files = _download_dataset(args.writingprompts_repo, allow_patterns=["data/train*.parquet"], target_root=cache_root)
    rocstories_files = _download_dataset(args.rocstories_repo, allow_patterns=["data/train*.parquet"], target_root=cache_root)
    if not writingprompts_files:
        raise FileNotFoundError(f"no writingprompts parquet files were downloaded from {args.writingprompts_repo}")
    if not rocstories_files:
        raise FileNotFoundError(f"no rocstories parquet files were downloaded from {args.rocstories_repo}")

    summary = build_creative_file(
        writingprompts=writingprompts_files,
        rocstories=rocstories_files,
        output=args.output,
        seed=args.seed,
        writingprompts_limit=args.writingprompts_limit,
        rocstories_limit=args.rocstories_limit,
    )
    print(summary)


if __name__ == "__main__":
    main()
