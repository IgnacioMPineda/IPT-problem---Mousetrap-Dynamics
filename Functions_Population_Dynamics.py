
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from Functions_final import distance_cells      
from Functions_final import final_history        
from Functions_final import kappa_exact


def regen_history(rho, f_k, p_succ_long, p_succ_short=0.9,
                  p_long=0.5, m=1, p_regen=0.0, L=50, steps=300, rng=None): # Setting the probabiloty of regeneration to zero 
                                                                            # returns the original model
    rng = rng or np.random.default_rng()
    state = np.where(rng.random((L, L)) < rho, 0, -1)
    ys, xs = np.where(state == 0)
    if len(xs) == 0:
        return [state.copy()]
    seed = rng.integers(len(xs))
    state[ys[seed], xs[seed]] = 1
    snapshots = [state.copy()]

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
                    k, p_hit = 1, p_succ_short
                candidates = distance_cells(fy, fx, k, L)
                if not candidates:
                    continue
                ty, tx = candidates[rng.integers(len(candidates))]
                if state[ty, tx] == 0 and rng.random() < p_hit:
                    new_fires.append((ty, tx))

        regen = (state == -2) & (rng.random((L, L)) < p_regen)   # NEW
        state[state == 1] = -2
        state[regen] = 0                                          # NEW

        if not new_fires:
            snapshots.append(state.copy())
            break
        for ty, tx in set(new_fires):
            state[ty, tx] = 1
        snapshots.append(state.copy())

    return snapshots

def q_invade(R, m=None):
    """Branching-process survival probability. m=None -> Poisson limit."""
    if R <= 1:
        return 0.0
    if m is None:
        g = lambda q: q - (1 - np.exp(-R * q))
    else:
        g = lambda q: q - (1 - (1 - R * q / m) ** m)
    return brentq(g, 1e-9, 1)

def invaded(history, n_c=50, t_c=15):
    last = history[-1]
    fired_total = np.sum(last == -2) + np.sum(last == 1)
    still_active = len(history) > t_c and np.sum(last == 1) > 0
    return fired_total >= n_c or still_active


def fixed_point(rho, kappa, p_regen):
    g = lambda i: i / (1 - np.exp(-kappa * i)) + i * (1 + 1 / p_regen) - rho
    i = brentq(g, 1e-12, rho)
    s = i / (1 - np.exp(-kappa * i))
    return s, i, i / p_regen

def jacobian_eigs(rho, kappa, p_regen):
    s, i, r = fixed_point(rho, kappa, p_regen)
    e = np.exp(-kappa * i)
    J = np.array([[e - p_regen,  -s * kappa * e - p_regen],
                  [1 - e,         s * kappa * e]])
    return np.linalg.eigvals(J)

def classify_run(history, t_burn=100, t_early=8):
    """Label one run and return the diagnostics."""
    i = np.array([np.sum(s == 1) for s in history]) / history[0].size
    T = len(i) - 1
    if i[-1] == 0:
        return dict(label='early extinction' if T < t_early else 'burnout', t_ext=T)
    if T <= t_burn + 20:
        raise ValueError("run too short: increase steps")

    x = i[t_burn:] - i[t_burn:].mean()
    acf = np.correlate(x, x, 'full')[len(x) - 1:]
    acf /= acf[0]
    freqs = np.fft.rfftfreq(len(x))
    power = np.abs(np.fft.rfft(x)) ** 2
    k = np.argmax(power[1:]) + 1
    half = len(x) // 2

    return dict(label='persist',
                mean_i=i[t_burn:].mean(),
                std_i=x.std(),
                amp_ratio=x[half:].std() / x[:half].std(),   # ~1 -> stationary
                acf_min=acf[1:40].min(),                     # < 0 -> oscillatory
                period=1 / freqs[k])


def late_means(rho, m_, p_regen, L=50, T=300, t_burn=100, n=3, rng=None):
    """Mean armed / firing fraction after the transient, over n persisting runs."""
    S, I = [], []
    while len(S) < n:
        h = regen_history(rho, f_k, p_succ_long, p_long=p_long, m=m_,
                          p_regen=p_regen, L=L, steps=T, rng=rng)
        if len(h) == T + 1 and np.any(h[-1] == 1):
            S.append(np.mean([np.mean(s == 0) for s in h[t_burn:]]))
            I.append(np.mean([np.mean(s == 1) for s in h[t_burn:]]))
    se = lambda x: np.std(x, ddof=1) / np.sqrt(len(x))
    return np.mean(S), se(S), np.mean(I), se(I)

def sirs_meanfield(rho, kappa, p_regen, i0, steps):
    # Mean-field SIRS map. Returns array (steps+1, 3) of s, i, r fractions.
    s, i, r = rho - i0, i0, 0.0
    out = [(s, i, r)]
    for _ in range(steps):
        new = s * (1 - np.exp(-kappa * i))
        s, i, r = s - new + p_regen * r, new, r + i - p_regen * r
        out.append((s, i, r))
    return np.array(out)

def trough_count(rho, kappa, p_regen, L, T=80, t_min=10):
    """Mean-field number of firing traps at the first trough."""
    mf = sirs_meanfield(rho, kappa, p_regen, i0=1 / L**2, steps=T)
    return mf[t_min:, 1].min() * L**2
