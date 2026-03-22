"""
Signal processing pipeline: generate → impair → DSP features → classify.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from backend.signal.generator import generate_signal
from backend.signal.impairments import apply_channel


@dataclass
class PipelineResult:
    signal_class: str
    snr_db: float
    raw_signal: NDArray[np.complex128]
    impaired_signal: NDArray[np.complex128]
    metadata: dict[str, Any] = field(default_factory=dict)


def run_pipeline(
    signal_class: str,
    n_samples: int = 4096,
    snr_db: float = 20.0,
    apply_impairments: bool = True,
    seed: int | None = None,
    **generator_kwargs,
) -> PipelineResult:
    """
    End-to-end pipeline:
      1. Generate clean signal
      2. Apply channel impairments (optional)
      3. Return PipelineResult with raw and impaired signals
    """
    rng = np.random.default_rng(seed)
    raw = generate_signal(signal_class, n_samples=n_samples, **generator_kwargs)

    if apply_impairments:
        impaired = apply_channel(raw, snr_db=snr_db, seed=int(rng.integers(0, 2**31)))
    else:
        impaired = raw.copy()

    power_dbfs = 10 * np.log10(np.mean(np.abs(raw) ** 2) + 1e-12)

    return PipelineResult(
        signal_class=signal_class,
        snr_db=snr_db,
        raw_signal=raw,
        impaired_signal=impaired,
        metadata={"power_dbfs": power_dbfs, "n_samples": n_samples},
    )


def batch_pipeline(
    classes: list[str],
    n_samples_each: int = 4096,
    snr_db_range: tuple[float, float] = (-20.0, 30.0),
    seed: int | None = None,
) -> list[PipelineResult]:
    """Generate one sample per class at random SNR from the given range."""
    rng = np.random.default_rng(seed)
    results = []
    for cls in classes:
        snr = float(rng.uniform(*snr_db_range))
        results.append(run_pipeline(cls, n_samples=n_samples_each, snr_db=snr,
                                    seed=int(rng.integers(0, 2**31))))
    return results
