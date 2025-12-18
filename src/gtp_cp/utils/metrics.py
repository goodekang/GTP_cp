from __future__ import annotations

import numpy as np


def concordance_index(event_time: np.ndarray, risk: np.ndarray, event_observed: np.ndarray) -> float:
    """
    Harrell's C-index.
    - Higher risk should correspond to shorter survival time.
    """
    t = np.asarray(event_time).astype(float)
    r = np.asarray(risk).astype(float)
    e = np.asarray(event_observed).astype(int)

    assert t.shape == r.shape == e.shape
    n = t.shape[0]

    concordant = 0.0
    permissible = 0.0
    ties = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            if t[i] == t[j]:
                continue
            # Determine which one had smaller time
            if t[i] < t[j]:
                i_early, j_late = i, j
            else:
                i_early, j_late = j, i

            # Pair is permissible only if earlier time observed event
            if e[i_early] == 0:
                continue

            permissible += 1.0
            if r[i_early] == r[j_late]:
                ties += 1.0
            elif r[i_early] > r[j_late]:
                concordant += 1.0

    if permissible == 0:
        return float("nan")
    return float((concordant + 0.5 * ties) / permissible)





