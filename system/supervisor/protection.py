"""
Supervisor Protection & Integrity Guardian (Layer A).
Enforces physical and cryptographic immutability of the supervisor directory.
The agent cannot modify, delete, disable, or overwrite any supervisor files.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class ProtectionError(PermissionError):
    """Raised when an unauthorized modification to supervisor files is detected or attempted."""
    pass


def compute_file_hash(filepath: Path) -> str:
    """Computes SHA-256 hash of a file with newline normalization (cross-platform LF)."""
    hasher = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            content = f.read()
        # Normalize CRLF to LF so hashes match on both Linux and Windows
        normalized = content.replace(b"\r\n", b"\n")
        hasher.update(normalized)
        return hasher.hexdigest()
    except Exception:
        return ""


def compute_supervisor_hashes(system_dir: Path) -> Dict[str, str]:
    """
    Computes SHA-256 hashes of all files in the system/ directory.
    Ignores volatile runtime files like .flag files.
    """
    hashes = {}
    system_dir = system_dir.resolve()
    if not system_dir.exists():
        return hashes

    for root, _, files in os.walk(system_dir):
        for file in sorted(files):
            # Skip flag files, temporary files, pycache, and the baseline hash file itself
            if file.endswith((".flag", ".tmp", ".pyc")) or file == "baseline_hashes.json" or "__pycache__" in root:
                continue
            full_path = Path(root) / file
            rel_path = full_path.relative_to(system_dir).as_posix()
            hashes[rel_path] = compute_file_hash(full_path)

    return hashes


def save_baseline_hashes(system_dir: Path, output_file: Path) -> Dict[str, str]:
    """Generates and writes baseline hashes of the supervisor directory."""
    hashes = compute_supervisor_hashes(system_dir)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2)
    return hashes


def verify_integrity(system_dir: Path, baseline: Dict[str, str]) -> Tuple[bool, List[str]]:
    """
    Verifies that the system/ directory has not been tampered with.
    Returns (is_valid, list_of_violations).
    """
    current_hashes = compute_supervisor_hashes(system_dir)
    violations = []

    # Check for modified or deleted files
    for rel_path, expected_hash in baseline.items():
        if rel_path not in current_hashes:
            violations.append(f"DELETED: system/{rel_path}")
        elif current_hashes[rel_path] != expected_hash:
            violations.append(f"MODIFIED: system/{rel_path}")

    # Check for newly added unexpected files
    for rel_path in current_hashes:
        if rel_path not in baseline:
            violations.append(f"UNAUTHORIZED_FILE_ADDED: system/{rel_path}")

    return (len(violations) == 0, violations)


def is_path_protected(target_path: Path, workspace_root: Path) -> bool:
    """
    Returns True if target_path falls inside the protected /system/ directory.
    Handles symlinks, relative traversal (..), and case differences.
    """
    try:
        resolved_workspace = workspace_root.resolve()
        resolved_target = (workspace_root / target_path).resolve()
        protected_dir = (resolved_workspace / "system").resolve()

        # Check if the target is within protected_dir
        return protected_dir == resolved_target or protected_dir in resolved_target.parents
    except Exception:
        # Default to safe: if path cannot be resolved, reject
        return True


def assert_path_allowed(target_path: Path, workspace_root: Path) -> None:
    """
    Raises ProtectionError if target_path attempts to access or modify protected supervisor files.
    """
    if is_path_protected(target_path, workspace_root):
        raise ProtectionError(
            f"SECURITY VIOLATION: Path '{target_path}' is inside the protected supervisor directory. "
            "The agent is strictly forbidden from accessing or modifying Layer A."
        )
