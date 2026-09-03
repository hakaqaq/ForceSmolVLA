"""Canonical source-closure identities used by development deployment bindings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from forcesmolvla.checkpoint import sha256_file


CLIENT_SOURCE_FILES = (
    "scripts/record_franka_hilserl_impedance.py",
    "scripts/hilserl_impedance_protocol.py",
    "scripts/record_franka_forcevla.py",
    "scripts/record_franka_forcevla_raw.py",
    "scripts/record_franka_spacemouse_publisher.py",
    "scripts/convert_franka_forcevla_raw_to_lerobot_v21.py",
)


def client_source_sha256(
    client_root: Path = Path("/home/rlc123/fr3_client_ws"),
) -> str:
    mapping = {
        relative: sha256_file(client_root / relative)
        for relative in CLIENT_SOURCE_FILES
    }
    return hashlib.sha256(
        json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
