"""
Protected Supervisor Package (Layer A).
"""

from .protection import compute_supervisor_hashes, verify_integrity, is_path_protected, assert_path_allowed, ProtectionError
from .watchdog import Watchdog, WatchdogStatus
from .health_check import HealthChecker, HealthStatus
from .supervisor import Supervisor

__all__ = [
    "Supervisor",
    "Watchdog",
    "WatchdogStatus",
    "HealthChecker",
    "HealthStatus",
    "compute_supervisor_hashes",
    "verify_integrity",
    "is_path_protected",
    "assert_path_allowed",
    "ProtectionError",
]
