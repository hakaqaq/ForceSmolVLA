#!/usr/bin/env python3
"""Canonical sampler-state comparison binding for the v5 boundary auditor."""

import hashlib

from forcesmolvla.rft.training_cycle import (
    SerializableReplacementSampler,
    SerializableUniqueSampler,
)
import audit_stage2b_long_run_recovery_boundary_v5 as auditor


def _install_comparable_state_dict(cls) -> None:
    original = cls.state_dict

    def comparable(self):
        result = original(self)
        state = result.pop("generator_state")
        result["generator_state_sha256"] = hashlib.sha256(
            state.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        return result

    cls.state_dict = comparable


_install_comparable_state_dict(SerializableUniqueSampler)
_install_comparable_state_dict(SerializableReplacementSampler)


if __name__ == "__main__":
    auditor.main()
