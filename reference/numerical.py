"""End-to-end mixed-state acceptance, separate from local FP32 primitive checks.

The coordinate budget grows linearly with residual depth: cast errors need not
be independent. A separate relative-energy gate applies to a complete layer's
hidden vector or cache, avoiding a condition-number test on a short near-zero head.
Original scalar/head diagnostics remain visible in native and Python reports.
"""
import numpy as np

POLICY = 'mixed-depth-rms-v3'


def state_errors(actual, reference, layer, mixed=True):
    actual, reference = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if actual.shape != reference.shape:
        raise ValueError('State tensor shapes differ')
    diff = np.abs(actual - reference)
    finite = np.isfinite(actual) & np.isfinite(reference)
    atol, rtol = (2e-3, 2e-2) if mixed else (1e-5, 1e-4)
    scale = ((layer + 1) * np.maximum(1, np.sqrt(np.mean(reference * reference, axis=-1, keepdims=True)))) if mixed else 1
    raw_failed = int(np.count_nonzero(~finite | (diff > atol + rtol * np.abs(reference))))
    failed = int(np.count_nonzero(~finite | (diff > atol * scale + rtol * np.abs(reference))))
    relative = float(np.linalg.norm(diff.ravel()) / max(1e-30, np.linalg.norm(reference.ravel())))
    return dict(raw_failed=raw_failed, failed=failed, nonfinite=int((~finite).sum()),
                max_abs=float(diff.max()), relative_l2=relative,
                passed=failed == 0 and (not mixed or relative <= .002))


def score_errors(actual, reference, layers):
    actual, reference = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    diff = np.abs(actual - reference)
    finite = np.isfinite(actual) & np.isfinite(reference)
    failed = int(np.count_nonzero(~finite | (diff > .002 * layers + .02 * np.abs(reference))))
    relative = float(np.linalg.norm(diff) / max(1e-30, np.linalg.norm(reference)))
    normalized = float(np.sqrt(np.mean(diff * diff)) / max(1., np.sqrt(np.mean(reference * reference))))
    return dict(failed=failed, max_abs=float(diff.max()), relative_l2=relative, normalized_rmse=normalized,
                passed=failed == 0 and normalized <= .002)
