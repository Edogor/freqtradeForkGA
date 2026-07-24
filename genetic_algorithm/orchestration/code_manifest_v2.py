"""Content-addressed Git/worktree provenance for reproducible V2 attempts."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

from pydantic import Field, model_validator

from genetic_algorithm.orchestration.artifact_store_v2 import canonical_config_hash
from genetic_algorithm.orchestration.result_contract import StrictV2Model


class CodeManifestError(ValueError):
    """Raised when repository provenance cannot be captured exactly."""


class UntrackedSourceFileV2(StrictV2Model):
    relative_path: str = Field(min_length=1)
    sha256: str = Field(min_length=64, max_length=64)
    size_bytes: int = Field(ge=0)


class CodeManifestV2(StrictV2Model):
    schema_version: str = "2.0"
    code_version: str = Field(min_length=1)
    tracked_state_sha256: str = Field(min_length=64, max_length=64)
    untracked_source_files: list[UntrackedSourceFileV2] = Field(default_factory=list)
    dirty: bool
    dirty_patch_hash: str | None = None

    @model_validator(mode="after")
    def _consistent_dirty_state(self) -> CodeManifestV2:
        paths = [item.relative_path for item in self.untracked_source_files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("untracked source files must be sorted and unique")
        expected_dirty = self.tracked_state_sha256 != _EMPTY_TRACKED_STATE_HASH or bool(paths)
        if self.dirty != expected_dirty:
            raise ValueError("dirty flag differs from captured repository state")
        expected_hash = (
            canonical_config_hash(
                {
                    "tracked_state_sha256": self.tracked_state_sha256,
                    "untracked_source_files": [
                        item.model_dump(mode="json") for item in self.untracked_source_files
                    ],
                }
            )
            if expected_dirty
            else None
        )
        if self.dirty_patch_hash != expected_hash:
            raise ValueError("dirty_patch_hash differs from captured repository state")
        return self


_SOURCE_SUFFIXES = {
    ".j2",
    ".jinja",
    ".py",
    ".pyi",
    ".sh",
    ".toml",
}
_EXCLUDED_PARTS = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__"}
_RUNTIME_SOURCE_PREFIXES = {
    ("genetic_algorithm", "data"),
    ("user_data", "strategies"),
}


def _digest_chunks(chunks: list[bytes]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(len(chunk).to_bytes(8, "big"))
        digest.update(chunk)
    return digest.hexdigest()


_EMPTY_TRACKED_STATE_HASH = _digest_chunks([b""])


def _git(root: Path, *arguments: str) -> bytes:
    try:
        process = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", b"")
        message = os.fsdecode(detail).strip() if detail else str(exc)
        raise CodeManifestError(f"cannot capture Git provenance: {message}") from exc
    return process.stdout


def _safe_repo_path(root: Path, raw_path: bytes) -> tuple[str, Path]:
    relative = os.fsdecode(raw_path)
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise CodeManifestError(f"Git returned unsafe repository path: {relative!r}")
    if not path.is_symlink() and root.resolve() not in path.resolve().parents:
        raise CodeManifestError(f"repository path escapes root: {relative!r}")
    return Path(relative).as_posix(), path


def _path_content(path: Path) -> bytes:
    if path.is_symlink():
        return b"symlink\0" + os.fsencode(path.readlink())
    if not path.exists():
        return b"deleted\0"
    if not path.is_file():
        raise CodeManifestError(f"unsupported changed repository entry: {path}")
    return b"file\0" + path.read_bytes()


def _tracked_state_hash(root: Path) -> str:
    raw_state = _git(root, "diff", "HEAD", "--raw", "--no-abbrev", "--no-ext-diff", "-z", "--")
    changed_paths = _git(root, "diff", "HEAD", "--name-only", "--no-ext-diff", "-z", "--")
    chunks = [raw_state]
    for raw_path in sorted(item for item in changed_paths.split(b"\0") if item):
        relative, path = _safe_repo_path(root, raw_path)
        chunks.extend([os.fsencode(relative), _path_content(path)])
    return _digest_chunks(chunks)


def _untracked_source_files(root: Path) -> list[UntrackedSourceFileV2]:
    raw_files = _git(root, "ls-files", "--others", "--exclude-standard", "-z", "--")
    snapshots = []
    for raw_path in sorted(item for item in raw_files.split(b"\0") if item):
        relative, path = _safe_repo_path(root, raw_path)
        relative_path = Path(relative)
        if any(part in _EXCLUDED_PARTS for part in relative_path.parts):
            continue
        if any(
            tuple(relative_path.parts[: len(prefix)]) == prefix
            for prefix in _RUNTIME_SOURCE_PREFIXES
        ):
            continue
        if relative_path.suffix.lower() not in _SOURCE_SUFFIXES:
            continue
        content = _path_content(path)
        snapshots.append(
            UntrackedSourceFileV2(
                relative_path=relative,
                sha256=hashlib.sha256(content).hexdigest(),
                size_bytes=path.lstat().st_size,
            )
        )
    return snapshots


def capture_code_manifest(repo_root: str | Path) -> CodeManifestV2:
    """Capture HEAD plus exact tracked changes and relevant untracked source files."""

    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise CodeManifestError(f"repository root does not exist: {root}")
    discovered_root = Path(
        os.fsdecode(_git(root, "rev-parse", "--show-toplevel")).strip()
    ).resolve()
    if discovered_root != root:
        raise CodeManifestError(f"repo_root must be the Git top level: {discovered_root}")
    code_version = os.fsdecode(_git(root, "rev-parse", "--verify", "HEAD")).strip()
    tracked_hash = _tracked_state_hash(root)
    untracked = _untracked_source_files(root)
    dirty = tracked_hash != _EMPTY_TRACKED_STATE_HASH or bool(untracked)
    dirty_hash = (
        canonical_config_hash(
            {
                "tracked_state_sha256": tracked_hash,
                "untracked_source_files": [item.model_dump(mode="json") for item in untracked],
            }
        )
        if dirty
        else None
    )
    return CodeManifestV2(
        code_version=code_version,
        tracked_state_sha256=tracked_hash,
        untracked_source_files=untracked,
        dirty=dirty,
        dirty_patch_hash=dirty_hash,
    )
