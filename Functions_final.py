"""Final 2D simulator functions used by Notebook 3 (extracted unchanged from Functions_for_Notebook_0.py)."""
import numpy as np


def distance_cells(y, x, k, L):  # row, column, distance, side length (square)
    """Cells on the Chebyshev ring at distance k from (y, x), clipped to the lattice."""
    cells = []
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            if max(abs(dy), abs(dx)) == k:
                ny, nx = y + dy, x + dx
                if 0 <= ny < L and 0 <= nx < L:
                    cells.append((ny, nx))
    return cells


def final_history(rho, f_k, p_succ_long, p_succ_short=0.9,
                  p_long=0.5, m=1, L=50, steps=200, rng=None):
    """List of lattice snapshots, one per generation."""
    rng = rng or np.random.default_rng()
    grid = (rng.random((L, L)) < rho).astype(int)
    state = np.where(grid == 1, 0, -1)
    ys, xs = np.where(grid == 1)
    if len(xs) == 0:
        return [state.copy()]
    seed = rng.integers(len(xs))
    state[ys[seed], xs[seed]] = 1

    snapshots = [state.copy()]          # save initial frame

    for t in range(steps):
        fired = np.argwhere(state == 1)
        if len(fired) == 0:
            break
        new_fires = []
        for fy, fx in fired:
            for _ in range(m):
                if rng.random() < p_long:
                    k = f_k(rng)
                    if k < 2:
                        continue
                    p_hit = p_succ_long(k)
                else:
                    k = 1
                    p_hit = p_succ_short
                candidates = distance_cells(fy, fx, k, L)
                if not candidates:
                    continue
                ty, tx = candidates[rng.integers(len(candidates))]
                if state[ty, tx] == 0 and rng.random() < p_hit:
                    new_fires.append((ty, tx))
        state[state == 1] = -2
        if not new_fires:
            snapshots.append(state.copy())
            break
        for (ty, tx) in set(new_fires):
            state[ty, tx] = 1
        snapshots.append(state.copy())   # save frame after this generation

    return snapshots


def pct_deactivated(history):
    """% of traps spent at each frame (inert cells excluded from the denominator)."""
    total = history[0].size - np.sum(history[0] == -1)
    return [100 * np.sum(s == -2) / total for s in history]


def dynamics_2d_toggle(rho, f_k, p_succ_long, p_succ_short=0.9, p_long=0.5,
                       m=1, L=50, steps=200,
                       use_short=True, use_long=True, rng=None):
    """
    2D counterpart of dynamics_1d_toggle. Short-range and long-range
    activation can each be switched on/off independently:
      use_short=True,  use_long=False -> short-range only
      use_short=False, use_long=True  -> long-range only
      use_short=True,  use_long=True  -> both, mixed via p_long
    """
    if not use_short and not use_long:
        raise ValueError("At least one of use_short / use_long must be True")

    rng = rng or np.random.default_rng()
    grid = (rng.random((L, L)) < rho).astype(int)
    state = np.where(grid == 1, 0, -1)
    ys, xs = np.where(grid == 1)
    if len(xs) == 0:
        return state
    seed = rng.integers(len(xs))
    state[ys[seed], xs[seed]] = 1

    if use_short and use_long:
        eff_p_long = p_long
    elif use_long:
        eff_p_long = 1.0
    else:
        eff_p_long = 0.0

    for t in range(steps):
        fired = np.argwhere(state == 1)
        if len(fired) == 0:
            break

        new_fires = []
        for fy, fx in fired:
            for _ in range(m):
                if rng.random() < eff_p_long:
                    k = f_k(rng)
                    if k < 2:
                        continue
                    p_hit = p_succ_long(k)
                else:
                    k = 1
                    p_hit = p_succ_short

                candidates = distance_cells(fy, fx, k, L)
                if not candidates:
                    continue
                ty, tx = candidates[rng.integers(len(candidates))]
                if state[ty, tx] == 0 and rng.random() < p_hit:
                    new_fires.append((ty, tx))

        state[state == 1] = -2

        if not new_fires:
            break
        for (ty, tx) in set(new_fires):
            state[ty, tx] = 1

    return state
