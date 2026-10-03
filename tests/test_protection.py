"""
Unit tests for Layer A Protection & Integrity Guardian.
"""

from pathlib import Path
import pytest

from system.supervisor.protection import (
    compute_supervisor_hashes,
    verify_integrity,
    is_path_protected,
    assert_path_allowed,
    ProtectionError,
)


def test_supervisor_hashes_computed():
    root = Path(__file__).resolve().parent.parent
    system_dir = root / "system"
    hashes = compute_supervisor_hashes(system_dir)
    assert len(hashes) > 0
    assert any("supervisor.py" in k for k in hashes)
    assert any("protection.py" in k for k in hashes)


def test_integrity_verification_pass():
    root = Path(__file__).resolve().parent.parent
    system_dir = root / "system"
    hashes = compute_supervisor_hashes(system_dir)
    valid, violations = verify_integrity(system_dir, hashes)
    assert valid is True
    assert len(violations) == 0


def test_integrity_verification_tamper_detected():
    root = Path(__file__).resolve().parent.parent
    system_dir = root / "system"
    hashes = compute_supervisor_hashes(system_dir)
    # Simulate altered baseline
    first_key = list(hashes.keys())[0]
    tampered_baseline = dict(hashes)
    tampered_baseline[first_key] = "0000000000000000000000000000000000000000000000000000000000000000"

    valid, violations = verify_integrity(system_dir, tampered_baseline)
    assert valid is False
    assert len(violations) == 1
    assert "MODIFIED" in violations[0]


def test_is_path_protected():
    root = Path(__file__).resolve().parent.parent
    # Paths inside system should be protected
    assert is_path_protected(Path("system/supervisor/supervisor.py"), root) is True
    assert is_path_protected(Path("system/kill_switch.flag"), root) is True
    assert is_path_protected(Path("agent/main.py"), root) is False
    assert is_path_protected(Path("project/src/ai_pipeline.py"), root) is False


def test_assert_path_allowed_raises_protection_error():
    root = Path(__file__).resolve().parent.parent
    with pytest.raises(ProtectionError) as exc_info:
        assert_path_allowed(Path("system/supervisor/supervisor.py"), root)
    assert "SECURITY VIOLATION" in str(exc_info.value)
