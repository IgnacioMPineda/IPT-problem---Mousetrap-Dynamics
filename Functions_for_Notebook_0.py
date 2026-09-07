import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



def simulate_MT_dynamics(rho, p0, kernel='exp', r0=3.0, L=50, steps=200, rng=None):
    rng = rng or np.random.default_rng()
    grid = (rng.random((L, L)) < rho).astype(int)      # 1 = armed trap present, 0 = empty
    state = np.where(grid == 1, 0, -1)                  # 0=armed, 1=fired, -1=no trap
    ys, xs = np.where(grid == 1)
    seed = rng.integers(len(xs))
    state[ys[seed], xs[seed]] = 1 

    # This block defines a grid where -1 means it is permamently inert
    #, 0 means it is armed, 1 means it fired this generation and -2 means it
    # previously fired

    def p_of_d(d):
        if kernel == 'exp': # This is a type of evolution that evolves exponentially, symbolizing a characteristic exponential decay
            return p0 * np.exp(-d / r0)
        elif kernel == 'inv': # This describes a second type of evolution where there is a distinct distance cutoff
            return np.minimum(p0 / np.maximum(d, 1), 1.0) * (d <= r0)

    # This block taked a numpy array of distances and converts it into a 
    # triggering probability

    for t in range(steps):
        fired = np.argwhere(state == 1)
        if len(fired) == 0:
            break #If there are no active firers, then the system will not change and the current timestep
            # will be outputted
        armed = np.argwhere(state == 0)
        if len(armed) == 0:
            break # Same concept as before, if all the cells have shot out, then 
            # the system will remain constant and the current timestep will be outputted
        for fy, fx in fired:
            d = np.hypot(armed[:,0]-fy, armed[:,1]-fx) # distance from armed cell to taget cell
            hits = rng.random(len(d)) < p_of_d(d) # just a probability
            for (ay, ax) in armed[hits]:
                state[ay, ax] = 1
        state[state == 1] = np.where(rng.random(np.sum(state==1)) < 1, 2, 1)  # mark spent
        state[state == 2] = -2  # spent, distinct from empty
    return state

def history_MT_dynamics(rho, p0, kernel='exp', r0=3.0, L=50, steps=200, rng=None):
    rng = rng or np.random.default_rng()
    grid = (rng.random((L, L)) < rho).astype(int)      # 1 = armed trap present, 0 = empty
    state = np.where(grid == 1, 0, -1)                  # 0=armed, 1=fired, -1=no trap
    ys, xs = np.where(grid == 1)
    seed = rng.integers(len(xs))
    state[ys[seed], xs[seed]] = 1 
    # This block defines a grid where -1 means it is permamently inert
    #, 0 means it is armed, 1 means it fired this generation and -2 means it
    # previously fired

    history = [state.copy()]  # record initial state as frame 0

    def p_of_d(d):
        if kernel == 'exp': # This is a type of evolution that evolves exponentially, symbolizinf a characteristic exponential decay
            return p0 * np.exp(-d / r0)
        elif kernel == 'inv': # This describes a second type of evolution where there is a distinct distance cutoff
            return np.minimum(p0 / np.maximum(d, 1), 1.0) * (d <= r0)
    # This block taked a numpy array of distances and converts it into a 
    # triggering probability

    for t in range(steps):
        fired = np.argwhere(state == 1)
        if len(fired) == 0:
            break #If there are no active firers, then the system will not change and the current timestep
            # will be outputted
        armed = np.argwhere(state == 0)
        if len(armed) == 0:
            break # Same concept as before, if all the cells have shot out, then 
            # the system will remain constant and the current timestep will be outputted
        newly_hit = np.zeros(len(armed), dtype=bool)
        for fy, fx in fired:
            d = np.hypot(armed[:,0]-fy, armed[:,1]-fx) # distance from armed cell to taget cell
            newly_hit |= rng.random(len(d)) < p_of_d(d) # just a probability
        for (ay, ax) in armed[newly_hit]:
            state[ay, ax] = 1
        # spend only the cells that were already fired BEFORE this step
        # (not the ones freshly hit above — they get to fire next round)
        for fy, fx in fired:
            state[fy, fx] = -2

        history.append(state.copy())  # record this generation

    return history


def show_timestep(history, t):
    """Display the grid state at timestep t from a history list."""
    if t < 0 or t >= len(history):
        raise ValueError(f"t={t} out of range; history has {len(history)} frames (0 to {len(history)-1})")

    state = history[t]

    # remap -1,0,1,-2 -> 0,1,2,3 for consistent coloring
    remap = {-1: 0, 0: 1, 1: 2, -2: 3}
    frame = np.vectorize(remap.get)(state)
    cmap = plt.matplotlib.colors.ListedColormap(
        ['#dddddd', '#f4a300', '#d62728', '#1f1f1f']  # inert, armed, firing, spent
    )

    fig, ax = plt.subplots()
    ax.imshow(frame, cmap=cmap, vmin=0, vmax=3)
    ax.set_title(f't = {t}')
    ax.axis('off')
    plt.show()

def animate_history(history, interval=150):
    remap = {-1: 0, 0: 1, 1: 2, -2: 3}
    cmap = plt.matplotlib.colors.ListedColormap(['#dddddd', '#f4a300', '#d62728', '#1f1f1f'])
    frames = [np.vectorize(remap.get)(f) for f in history]
    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], cmap=cmap, vmin=0, vmax=3)
    ax.axis('off')
    def update(i):
        im.set_data(frames[i])
        ax.set_title(f't={i}')
        return [im]
    return FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False)


def plot_alive(history):
    """
    Plots counts of armed, firing, and spent cells at each timestep.
    'Alive' = armed (0) + firing (1); adjust as needed.
    """
    armed_counts  = [np.sum(s == 0)  for s in history]
    firing_counts = [np.sum(s == 1)  for s in history]
    spent_counts  = [np.sum(s == -2) for s in history]
    alive_counts  = [a + f for a, f in zip(armed_counts, firing_counts)]

    t = np.arange(len(history))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, alive_counts, label='alive (armed + firing)', color='tab:green', lw=2)
    ax.plot(t, armed_counts, label='armed', color='tab:orange', ls='--')
    ax.plot(t, firing_counts, label='firing', color='tab:red', ls='--')
    ax.plot(t, spent_counts, label='spent', color='black', ls=':')
    ax.set_xlabel('timestep')
    ax.set_ylabel('cell count')
    ax.set_title('Population dynamics over time')
    ax.legend()
    plt.tight_layout()
    plt.show()

def p_ball_of_d(d, p0_ball, d_opt, r_rise, r_fall): # Long range kernel
    d = np.asarray(d, dtype=float)
    rising = d <= d_opt
    return np.where(
        rising,
        p0_ball * np.exp(-(d_opt - d) / r_rise),   # 0 -> p0_ball as d -> d_opt
        p0_ball * np.exp(-(d - d_opt) / r_fall)     # p0_ball -> 0 as d grows past d_opt
    )

def p_trap_of_d(d, p0_trap, r0_trap): # Short range kernel
    return p0_trap * np.exp(-d / r0_trap)

def _combined_p_of_d(d, p0_trap, r0_trap, p0_ball, d_opt, r_rise, r_fall): # Calculates the probability associated
    # with both kernels
    p_trap = p_trap_of_d(d, p0_trap, r0_trap)
    p_ball = p_ball_of_d(d, p0_ball, d_opt, r_rise, r_fall)
    return 1 - (1 - p_trap) * (1 - p_ball)


def simulate_MT_dynamics_v2(rho, p0_trap, r0_trap,
                             p0_ball, d_opt, r_rise, r_fall,
                             L=50, steps=200, rng=None): # This is the original dynamics function but the probability is now
    # the conjoined probability
    rng = rng or np.random.default_rng()
    grid = (rng.random((L, L)) < rho).astype(int)
    state = np.where(grid == 1, 0, -1)
    ys, xs = np.where(grid == 1)
    seed = rng.integers(len(xs))
    state[ys[seed], xs[seed]] = 1

    for t in range(steps):
        fired = np.argwhere(state == 1)
        if len(fired) == 0:
            break
        armed = np.argwhere(state == 0)
        if len(armed) == 0:
            break

        newly_hit = np.zeros(len(armed), dtype=bool)
        for fy, fx in fired:
            d = np.hypot(armed[:, 0] - fy, armed[:, 1] - fx)
            p_total = _combined_p_of_d(d, p0_trap, r0_trap, p0_ball, d_opt, r_rise, r_fall)
            newly_hit |= rng.random(len(d)) < p_total

        for (ay, ax) in armed[newly_hit]:
            state[ay, ax] = 1
        for fy, fx in fired:
            state[fy, fx] = -2

    return state


def history_MT_dynamics_v2(rho, p0_trap, r0_trap,
                            p0_ball, d_opt, r_rise, r_fall,
                            L=50, steps=200, rng=None):
    rng = rng or np.random.default_rng()
    grid = (rng.random((L, L)) < rho).astype(int)
    state = np.where(grid == 1, 0, -1)
    ys, xs = np.where(grid == 1)
    seed = rng.integers(len(xs))
    state[ys[seed], xs[seed]] = 1

    history = [state.copy()]

    for t in range(steps):
        fired = np.argwhere(state == 1)
        if len(fired) == 0:
            break
        armed = np.argwhere(state == 0)
        if len(armed) == 0:
            break

        newly_hit = np.zeros(len(armed), dtype=bool)
        for fy, fx in fired:
            d = np.hypot(armed[:, 0] - fy, armed[:, 1] - fx)
            p_total = _combined_p_of_d(d, p0_trap, r0_trap, p0_ball, d_opt, r_rise, r_fall)
            newly_hit |= rng.random(len(d)) < p_total

        for (ay, ax) in armed[newly_hit]:
            state[ay, ax] = 1
        for fy, fx in fired:
            state[fy, fx] = -2

        history.append(state.copy())

    return history

def short_term_interaction(rho, p0, kernel='exp', r0=3.0, L=50, steps=200, rng=None):
    rng = rng or np.random.default_rng()
    grid = (rng.random((L, L)) < rho).astype(int)      # 1 = armed trap present, 0 = empty
    state = np.where(grid == 1, 0, -1)                  # 0=armed, 1=fired, -1=no trap
    ys, xs = np.where(grid == 1)
    seed = rng.integers(len(xs))
    state[ys[seed], xs[seed]] = 1 

    # This block defines a grid where -1 means it is permamently inert
    #, 0 means it is armed, 1 means it fired this generation and -2 means it
    # previously fired

    def p_of_d(d):
        if kernel == 'exp': # This is a type of evolution that evolves exponentially, symbolizing a characteristic exponential decay
            return p0 * np.exp(-d / r0)
        elif kernel == 'inv': # This describes a second type of evolution where there is a distinct distance cutoff
            return np.minimum(p0 / np.maximum(d, 1), 1.0) * (d <= r0)

    # This block taked a numpy array of distances and converts it into a 
    # triggering probability

    for t in range(steps):
        fired = np.argwhere(state == 1)
        if len(fired) == 0:
            break #If there are no active firers, then the system will not change and the current timestep
            # will be outputted
        armed = np.argwhere(state == 0)
        if len(armed) == 0:
            break # Same concept as before, if all the cells have shot out, then 
            # the system will remain constant and the current timestep will be outputted
        for fy, fx in fired:
            d = np.hypot(armed[:,0]-fy, armed[:,1]-fx) # distance from armed cell to taget cell
            hits = rng.random(len(d)) < p_of_d(d) # just a probability
            for (ay, ax) in armed[hits]:
                state[ay, ax] = 1
        state[state == 1] = np.where(rng.random(np.sum(state==1)) < 1, 2, 1)  # mark spent
        state[state == 2] = -2  # spent, distinct from empty
    return state


def long_term_interaction(rho, p0_ball, d_opt, r_rise, r_fall, L=50, steps=200, rng=None):
    rng = rng or np.random.default_rng()
    grid = (rng.random((L, L)) < rho).astype(int)     
    state = np.where(grid == 1, 0, -1)                  
    ys, xs = np.where(grid == 1)
    seed = rng.integers(len(xs))
    state[ys[seed], xs[seed]] = 1


    for t in range(steps):
        fired = np.argwhere(state == 1)
        if len(fired) == 0:
            break 
        armed = np.argwhere(state == 0)
        if len(armed) == 0:
            break  
        for fy, fx in fired:
            d = np.hypot(armed[:, 0] - fy, armed[:, 1] - fx)  # distance from armed cell to target cell
            hits = rng.random(len(d)) < p_ball_of_d(d, p0_ball, d_opt, r_rise, r_fall)  # long-term-only probability
            for (ay, ax) in armed[hits]:
                state[ay, ax] = 1
        state[state == 1] = np.where(rng.random(np.sum(state == 1)) < 1, 2, 1)  # mark spent
        state[state == 2] = -2  
    return state

def plot_deactivated_vs_density_time(rho_values, p0_trap, r0_trap, p0_ball, d_opt, r_rise, r_fall,
                                      L=50, steps=200, rng=None):
    rng = rng or np.random.default_rng()
    fig, ax = plt.subplots(figsize=(7, 4))

    for rho in rho_values:
        hist = history_MT_dynamics_v2(rho, p0_trap, r0_trap, p0_ball, d_opt, r_rise, r_fall,
                                       L=L, steps=steps, rng=rng)
        pct_deactivated = [100 * np.sum(s == -2) / (L * L) for s in hist]
        ax.plot(pct_deactivated, label=f'ρ={rho:.2f}')

    ax.set_xlabel('timestep')
    ax.set_ylabel('% cells deactivated')
    ax.set_title('Deactivation over time vs density (conjoined)')
    ax.legend()
    plt.tight_layout()
    plt.show()

def p_ball_of_d(d, p0_ball, d_opt, r_rise, r_fall):
    d = np.asarray(d, dtype=float)
    rising = d <= d_opt
    return np.where(
        rising,
        p0_ball * np.exp(-(d_opt - d) / r_rise),
        p0_ball * np.exp(-(d - d_opt) / r_fall)
    )

def p_trap_of_d(d, p0_trap, r0_trap):
    return p0_trap * np.exp(-d / r0_trap)

def _combined_p_of_d(d, p0_trap, r0_trap, p0_ball, d_opt, r_rise, r_fall):
    p_trap = p_trap_of_d(d, p0_trap, r0_trap)
    p_ball = p_ball_of_d(d, p0_ball, d_opt, r_rise, r_fall)
    return 1 - (1 - p_trap) * (1 - p_ball)

def _init_grid(rho, L, rng): # Builds the starting state for one simulation run
    grid = (rng.random((L, L)) < rho).astype(int)
    state = np.where(grid == 1, 0, -1)
    ys, xs = np.where(grid == 1)
    seed = rng.integers(len(xs))
    state[ys[seed], xs[seed]] = 1
    return state

def history_short(rho, p0_trap, r0_trap, L=50, steps=200, rng=None):
    rng = rng or np.random.default_rng()
    state = _init_grid(rho, L, rng)
    history = [state.copy()]
    for t in range(steps):
        fired = np.argwhere(state == 1)
        if len(fired) == 0: break
        armed = np.argwhere(state == 0)
        if len(armed) == 0: break
        newly_hit = np.zeros(len(armed), dtype=bool)
        for fy, fx in fired:
            d = np.hypot(armed[:, 0] - fy, armed[:, 1] - fx)
            newly_hit |= rng.random(len(d)) < p_trap_of_d(d, p0_trap, r0_trap)
        for (ay, ax) in armed[newly_hit]:
            state[ay, ax] = 1
        for fy, fx in fired:
            state[fy, fx] = -2
        history.append(state.copy())
    return history

def history_long(rho, p0_ball, d_opt, r_rise, r_fall, L=50, steps=200, rng=None):
    rng = rng or np.random.default_rng()
    state = _init_grid(rho, L, rng)
    history = [state.copy()]
    for t in range(steps):
        fired = np.argwhere(state == 1)
        if len(fired) == 0: break
        armed = np.argwhere(state == 0)
        if len(armed) == 0: break
        newly_hit = np.zeros(len(armed), dtype=bool)
        for fy, fx in fired:
            d = np.hypot(armed[:, 0] - fy, armed[:, 1] - fx)
            newly_hit |= rng.random(len(d)) < p_ball_of_d(d, p0_ball, d_opt, r_rise, r_fall)
        for (ay, ax) in armed[newly_hit]:
            state[ay, ax] = 1
        for fy, fx in fired:
            state[fy, fx] = -2
        history.append(state.copy())
    return history

def history_conjoined(rho, p0_trap, r0_trap, p0_ball, d_opt, r_rise, r_fall, L=50, steps=200, rng=None):
    rng = rng or np.random.default_rng()
    state = _init_grid(rho, L, rng)
    history = [state.copy()]
    for t in range(steps):
        fired = np.argwhere(state == 1)
        if len(fired) == 0: break
        armed = np.argwhere(state == 0)
        if len(armed) == 0: break
        newly_hit = np.zeros(len(armed), dtype=bool)
        for fy, fx in fired:
            d = np.hypot(armed[:, 0] - fy, armed[:, 1] - fx)
            p_total = _combined_p_of_d(d, p0_trap, r0_trap, p0_ball, d_opt, r_rise, r_fall)
            newly_hit |= rng.random(len(d)) < p_total
        for (ay, ax) in armed[newly_hit]:
            state[ay, ax] = 1
        for fy, fx in fired:
            state[fy, fx] = -2
        history.append(state.copy())
    return history

def pct_deactivated(history, L):
    return [100 * np.sum(s == -2) / (L * L) for s in history] # To obtain teh y-axis values for the plot

def plot_all_settings_vs_density_time(rho_values, p0_trap, r0_trap, p0_ball, d_opt, r_rise, r_fall,
                                       L=50, steps=200, seed=0):
    n = len(rho_values)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.2*nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for i, rho in enumerate(rho_values):
        ax = axes[i]
        rng_s = np.random.default_rng(seed)
        rng_l = np.random.default_rng(seed)
        rng_c = np.random.default_rng(seed)

        h_short = history_short(rho, p0_trap, r0_trap, L=L, steps=steps, rng=rng_s)
        h_long  = history_long(rho, p0_ball, d_opt, r_rise, r_fall, L=L, steps=steps, rng=rng_l)
        h_conj  = history_conjoined(rho, p0_trap, r0_trap, p0_ball, d_opt, r_rise, r_fall,
                                     L=L, steps=steps, rng=rng_c)

        ax.plot(pct_deactivated(h_short, L), label='short', color='tab:orange')
        ax.plot(pct_deactivated(h_long, L),  label='long',  color='tab:blue')
        ax.plot(pct_deactivated(h_conj, L),  label='conjoined', color='tab:green')
        ax.set_title(f'ρ={rho:.2f}')

    for ax in axes[n:]:
        ax.axis('off')

    axes[0].legend(fontsize=8)
    fig.supxlabel('timestep')
    fig.supylabel('% cells deactivated')
    fig.suptitle('Deactivation over time: short vs long vs conjoined, by density')
    plt.tight_layout()
    plt.show()

def _init_grid_random(rho, p_active, L, rng):
    grid = (rng.random((L, L)) < rho).astype(int)          # 1 = trap present
    state = np.where(grid == 1, 0, -1)                      # 0=armed, -1=no trap
    trap_mask = grid == 1
    active_mask = trap_mask & (rng.random((L, L)) < p_active)  # each trap independently active w.p. p_active
    state[active_mask] = 1
    return state


def simulate_dynamics(rho, p_active, p_of_d_func, L=50, steps=200, rng=None):
    def _init_grid_random(rho, p_active, L, rng):
        grid = (rng.random((L, L)) < rho).astype(int)          # 1 = trap present
        state = np.where(grid == 1, 0, -1)                      # 0=armed, -1=no trap
        trap_mask = grid == 1
        active_mask = trap_mask & (rng.random((L, L)) < p_active)  # each trap independently active w.p. p_active
        state[active_mask] = 1
        return state
    
    rng = rng or np.random.default_rng()
    state = _init_grid_random(rho, p_active, L, rng)

    for t in range(steps):
        fired = np.argwhere(state == 1)
        if len(fired) == 0:
            break
        armed = np.argwhere(state == 0)
        if len(armed) == 0:
            break

        newly_hit = np.zeros(len(armed), dtype=bool)
        for fy, fx in fired:
            d = np.hypot(armed[:, 0] - fy, armed[:, 1] - fx)
            newly_hit |= rng.random(len(d)) < p_of_d_func(d)
        for (ay, ax) in armed[newly_hit]:
            state[ay, ax] = 1
        for fy, fx in fired:
            state[fy, fx] = -2

    return state


def history_dynamics(rho, p_active, p_of_d_func, L=50, steps=200, rng=None):
    def _init_grid_random(rho, p_active, L, rng):
        grid = (rng.random((L, L)) < rho).astype(int)          # 1 = trap present
        state = np.where(grid == 1, 0, -1)                      # 0=armed, -1=no trap
        trap_mask = grid == 1
        active_mask = trap_mask & (rng.random((L, L)) < p_active)  # each trap independently active w.p. p_active
        state[active_mask] = 1
        return state
    rng = rng or np.random.default_rng()
    state = _init_grid_random(rho, p_active, L, rng)
    history = [state.copy()]

    for t in range(steps):
        fired = np.argwhere(state == 1)
        if len(fired) == 0:
            break
        armed = np.argwhere(state == 0)
        if len(armed) == 0:
            break

        newly_hit = np.zeros(len(armed), dtype=bool)
        for fy, fx in fired:
            d = np.hypot(armed[:, 0] - fy, armed[:, 1] - fx)
            newly_hit |= rng.random(len(d)) < p_of_d_func(d)
        for (ay, ax) in armed[newly_hit]:
            state[ay, ax] = 1
        for fy, fx in fired:
            state[fy, fx] = -2

        history.append(state.copy())

    return history


def p_ball_of_d(d, p0_ball, d_opt, r_rise, r_fall):  # Long range kernel (from your notebook)
    d = np.asarray(d, dtype=float)
    rising = d <= d_opt
    return np.where(
        rising,
        p0_ball * np.exp(-(d_opt - d) / r_rise),   # 0 -> p0_ball as d -> d_opt
        p0_ball * np.exp(-(d - d_opt) / r_fall)    # p0_ball -> 0 as d grows past d_opt
    )


def short_and_long_term_interaction_neighbors(rho, p, p0_ball, d_opt, r_rise, r_fall,
                                               L=50, steps=200, neighborhood='moore', rng=None):
    rng = rng or np.random.default_rng()
    grid = (rng.random((L, L)) < rho).astype(int)
    state = np.where(grid == 1, 0, -1)
    ys, xs = np.where(grid == 1)
    seed = rng.integers(len(xs))
    state[ys[seed], xs[seed]] = 1

    if neighborhood == 'moore':
        offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if not (dy == 0 and dx == 0)]
    elif neighborhood == 'von_neumann':
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        raise ValueError("neighborhood must be 'moore' or 'von_neumann'")

    for t in range(steps):
        fired = np.argwhere(state == 1)
        if len(fired) == 0:
            break
        armed = np.argwhere(state == 0)
        if len(armed) == 0:
            break

        newly_hit_long = np.zeros(len(armed), dtype=bool)   # indexed against `armed`
        newly_hit_short = set()                              # explicit (y, x) coords

        for fy, fx in fired:
            # --- long-range: any armed cell, probability from the distance kernel ---
            d = np.hypot(armed[:, 0] - fy, armed[:, 1] - fx)
            newly_hit_long |= rng.random(len(d)) < p_ball_of_d(d, p0_ball, d_opt, r_rise, r_fall)

            # --- short-range: only immediate neighbors, flat input probability p ---
            for dy, dx in offsets:
                ny, nx = fy + dy, fx + dx
                if 0 <= ny < L and 0 <= nx < L and state[ny, nx] == 0:
                    if rng.random() < p:
                        newly_hit_short.add((ny, nx))

        for (ay, ax) in armed[newly_hit_long]:
            state[ay, ax] = 1
        for (ny, nx) in newly_hit_short:
            state[ny, nx] = 1

        for fy, fx in fired:
            state[fy, fx] = -2

    return state


def history_short_and_long_term_interaction_neighbors(rho, p, p0_ball, d_opt, r_rise, r_fall,
                                                        L=50, steps=200, neighborhood='moore', rng=None):
    rng = rng or np.random.default_rng()
    grid = (rng.random((L, L)) < rho).astype(int)
    state = np.where(grid == 1, 0, -1)
    ys, xs = np.where(grid == 1)
    seed = rng.integers(len(xs))
    state[ys[seed], xs[seed]] = 1

    history = [state.copy()]

    if neighborhood == 'moore':
        offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if not (dy == 0 and dx == 0)]
    elif neighborhood == 'von_neumann':
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        raise ValueError("neighborhood must be 'moore' or 'von_neumann'")

    for t in range(steps):
        fired = np.argwhere(state == 1)
        if len(fired) == 0:
            break
        armed = np.argwhere(state == 0)
        if len(armed) == 0:
            break

        newly_hit_long = np.zeros(len(armed), dtype=bool)
        newly_hit_short = set()

        for fy, fx in fired:
            d = np.hypot(armed[:, 0] - fy, armed[:, 1] - fx)
            newly_hit_long |= rng.random(len(d)) < p_ball_of_d(d, p0_ball, d_opt, r_rise, r_fall)

            for dy, dx in offsets:
                ny, nx = fy + dy, fx + dx
                if 0 <= ny < L and 0 <= nx < L and state[ny, nx] == 0:
                    if rng.random() < p:
                        newly_hit_short.add((ny, nx))

        for (ay, ax) in armed[newly_hit_long]:
            state[ay, ax] = 1
        for (ny, nx) in newly_hit_short:
            state[ny, nx] = 1

        for fy, fx in fired:
            state[fy, fx] = -2   # spend only cells that fired BEFORE this step

        history.append(state.copy())

    return history

def short_term_interaction_neighbors(rho, p, L=50, steps=200, neighborhood='moore', rng=None):

    rng = rng or np.random.default_rng()
    grid = (rng.random((L, L)) < rho).astype(int)     
    state = np.where(grid == 1, 0, -1)                 
    ys, xs = np.where(grid == 1)
    seed = rng.integers(len(xs))
    state[ys[seed], xs[seed]] = 1                     

    if neighborhood == 'moore':
        offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if not (dy == 0 and dx == 0)]
    elif neighborhood == 'von_neumann':
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        raise ValueError("neighborhood must be 'moore' or 'von_neumann'")

    for t in range(steps):                           
        fired = np.argwhere(state == 1)

        newly_hit = set()
        for fy, fx in fired:
            for dy, dx in offsets:                   
                ny, nx = fy + dy, fx + dx
                if 0 <= ny < L and 0 <= nx < L and state[ny, nx] == 0:
                    if rng.random() < p:              
                        newly_hit.add((ny, nx))

        for ny, nx in newly_hit:
            state[ny, nx] = 1                          
        for fy, fx in fired:
            state[fy, fx] = -2                        

    return state


def history_short_term_interaction_neighbors(rho, p, L=50, steps=200, neighborhood='moore', rng=None):

    rng = rng or np.random.default_rng()
    grid = (rng.random((L, L)) < rho).astype(int)     
    state = np.where(grid == 1, 0, -1)               
    ys, xs = np.where(grid == 1)
    seed = rng.integers(len(xs))
    state[ys[seed], xs[seed]] = 1

    history = [state.copy()]                            

    if neighborhood == 'moore':
        offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if not (dy == 0 and dx == 0)]
    elif neighborhood == 'von_neumann':
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        raise ValueError("neighborhood must be 'moore' or 'von_neumann'")

    for t in range(steps):                            
        fired = np.argwhere(state == 1)

        newly_hit = set()
        for fy, fx in fired:
            for dy, dx in offsets:
                ny, nx = fy + dy, fx + dx
                if 0 <= ny < L and 0 <= nx < L and state[ny, nx] == 0:
                    if rng.random() < p:
                        newly_hit.add((ny, nx))

        for ny, nx in newly_hit:
            state[ny, nx] = 1
        for fy, fx in fired:
            state[fy, fx] = -2                          

        history.append(state.copy())                   

    return history

def p_ball_of_d(d, p0_ball, d_opt, r_rise, r_fall):
    d = np.asarray(d, dtype=float)
    rising = d <= d_opt
    return np.where(
        rising,
        p0_ball * np.exp(-(d_opt - d) / r_rise),
        p0_ball * np.exp(-(d - d_opt) / r_fall)
    )

def _init_grid(rho, L, rng):  # Builds the starting state for one simulation run
    grid = (rng.random((L, L)) < rho).astype(int)
    state = np.where(grid == 1, 0, -1)
    ys, xs = np.where(grid == 1)
    seed = rng.integers(len(xs))
    state[ys[seed], xs[seed]] = 1
    return state

def _neighbor_offsets(neighborhood):
    if neighborhood == 'moore':
        return [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if not (dy == 0 and dx == 0)]
    elif neighborhood == 'von_neumann':
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        raise ValueError("neighborhood must be 'moore' or 'von_neumann'")

def history_short(rho, p, L=50, steps=200, neighborhood='moore', rng=None):

    rng = rng or np.random.default_rng()
    state = _init_grid(rho, L, rng)
    history = [state.copy()]
    offsets = _neighbor_offsets(neighborhood)

    for t in range(steps):
        fired = np.argwhere(state == 1)
        newly_hit = set()
        for fy, fx in fired:
            for dy, dx in offsets:
                ny, nx = fy + dy, fx + dx
                if 0 <= ny < L and 0 <= nx < L and state[ny, nx] == 0:
                    if rng.random() < p:
                        newly_hit.add((ny, nx))
        for ny, nx in newly_hit:
            state[ny, nx] = 1
        for fy, fx in fired:
            state[fy, fx] = -2
        history.append(state.copy())
    return history

def history_long(rho, p0_ball, d_opt, r_rise, r_fall, L=50, steps=200, rng=None):
   
    rng = rng or np.random.default_rng()
    state = _init_grid(rho, L, rng)
    history = [state.copy()]
    for t in range(steps):
        fired = np.argwhere(state == 1)
        if len(fired) == 0: break
        armed = np.argwhere(state == 0)
        if len(armed) == 0: break
        newly_hit = np.zeros(len(armed), dtype=bool)
        for fy, fx in fired:
            d = np.hypot(armed[:, 0] - fy, armed[:, 1] - fx)
            newly_hit |= rng.random(len(d)) < p_ball_of_d(d, p0_ball, d_opt, r_rise, r_fall)
        for (ay, ax) in armed[newly_hit]:
            state[ay, ax] = 1
        for fy, fx in fired:
            state[fy, fx] = -2
        history.append(state.copy())
    return history

def history_conjoined(rho, p, p0_ball, d_opt, r_rise, r_fall, L=50, steps=200, neighborhood='moore', rng=None):

    rng = rng or np.random.default_rng()
    state = _init_grid(rho, L, rng)
    history = [state.copy()]
    offsets = _neighbor_offsets(neighborhood)

    for t in range(steps):
        fired = np.argwhere(state == 1)
        armed = np.argwhere(state == 0)

        newly_hit_long = np.zeros(len(armed), dtype=bool)
        newly_hit_short = set()

        for fy, fx in fired:
            d = np.hypot(armed[:, 0] - fy, armed[:, 1] - fx)
            newly_hit_long |= rng.random(len(d)) < p_ball_of_d(d, p0_ball, d_opt, r_rise, r_fall)

            for dy, dx in offsets:
                ny, nx = fy + dy, fx + dx
                if 0 <= ny < L and 0 <= nx < L and state[ny, nx] == 0:
                    if rng.random() < p:
                        newly_hit_short.add((ny, nx))

        for (ay, ax) in armed[newly_hit_long]:
            state[ay, ax] = 1
        for (ny, nx) in newly_hit_short:
            state[ny, nx] = 1
        for fy, fx in fired:
            state[fy, fx] = -2

        history.append(state.copy())
    return history

def pct_deactivated(history, L):
    return [100 * np.sum(s == -2) / (L * L) for s in history]  # To obtain the y-axis values for the plot

def plot_all_settings_vs_density_time(rho_values, p, p0_ball, d_opt, r_rise, r_fall,
                                       L=50, steps=200, neighborhood='moore', seed=0):
    n = len(rho_values)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.2*nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for i, rho in enumerate(rho_values):
        ax = axes[i]
        rng_s = np.random.default_rng(seed)
        rng_l = np.random.default_rng(seed)
        rng_c = np.random.default_rng(seed)

        h_short = history_short(rho, p, L=L, steps=steps, neighborhood=neighborhood, rng=rng_s)
        h_long  = history_long(rho, p0_ball, d_opt, r_rise, r_fall, L=L, steps=steps, rng=rng_l)
        h_conj  = history_conjoined(rho, p, p0_ball, d_opt, r_rise, r_fall,
                                     L=L, steps=steps, neighborhood=neighborhood, rng=rng_c)

        ax.plot(pct_deactivated(h_short, L), label='short', color='tab:orange')
        ax.plot(pct_deactivated(h_long, L),  label='long',  color='tab:blue')
        ax.plot(pct_deactivated(h_conj, L),  label='conjoined', color='tab:green')
        ax.set_title(f'ρ={rho:.2f}')

    for ax in axes[n:]:
        ax.axis('off')

    axes[0].legend(fontsize=8)
    fig.supxlabel('timestep')
    fig.supylabel('% cells deactivated')
    fig.suptitle('Deactivation over time: short vs long vs conjoined, by density')
    plt.tight_layout()
    plt.show()

def pct_deactivated(history, L):
    return [100 * (np.sum(s == -2) + np.sum(s == -1)) / (L * L) for s in history]

def distance_cells(y, x, k, L): # column, row, distance, side length (square)
    
    cells = []
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            if max(abs(dy), abs(dx)) == k:
                ny, nx = y + dy, x + dx
                if 0 <= ny < L and 0 <= nx < L:
                    cells.append((ny, nx))

    return cells


def simulate_longrange(rho, f_k, p_succ, m=1, L=50, steps=200, rng=None):
    """
    Here, we define the activation mechanism that will later be integrated into the dynamics function.
    """
    rng = rng or np.random.default_rng()

    grid = (rng.random((L, L)) < rho).astype(int)
    state = np.where(grid == 1, 0, -1)

    ys, xs = np.where(grid == 1)
    if len(xs) == 0:
        return state, []
    seed = rng.integers(len(xs))
    state[ys[seed], xs[seed]] = 1

    history = [1]

    for t in range(steps):
        fired = np.argwhere(state == 1)
        if len(fired) == 0:
            break

        new_fires = []
        for fy, fx in fired:
            for _ in range(m):                      # m independent balls per source
                k = f_k(rng)
                if k < 1:
                    continue
                candidates = distance_cells(fy, fx, k, L)
                if not candidates:
                    continue
                ty, tx = candidates[rng.integers(len(candidates))]
                if state[ty, tx] == 0 and rng.random() < p_succ(k):
                    new_fires.append((ty, tx))

        state[state == 1] = -2                       # retire this generation's sources

        if not new_fires:
            break
        for (ty, tx) in set(new_fires):               # deduce multi-hits same generation
            state[ty, tx] = 1
        history.append(len(set(new_fires)))

    return state, history


def final_dynamics(rho, f_k, p_succ_long, p_succ_short=0.9, p_long=0.5,
                       m=1, L=50, steps=200, rng=None):
    """
    This finalized dynamics function now integrates the short and new long term dynamics to most accurately represent mousetrap dynamics
    """
    rng = rng or np.random.default_rng()

    grid = (rng.random((L, L)) < rho).astype(int)
    state = np.where(grid == 1, 0, -1)

    ys, xs = np.where(grid == 1)
    if len(xs) == 0:
        return state, []
    seed = rng.integers(len(xs))
    state[ys[seed], xs[seed]] = 1

    history = [1]

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
            break
        for (ty, tx) in set(new_fires):
            state[ty, tx] = 1
        history.append(len(set(new_fires)))

    return state, history


def final_history(n_trials, rho, f_k, p_succ_long, p_succ_short=0.9, p_long=0.5,
                m=1, L=50, steps=200, rng=None):
    """
    This is the finalized version of the history function
    """
    rng = rng or np.random.default_rng()
    histories, fractions_fired = [], []

    for _ in range(n_trials):
        state, history = final_dynamics(
            rho, f_k, p_succ_long, p_succ_short, p_long, m, L, steps, rng)
        histories.append(history)
        n_traps = np.sum(state != -1)
        n_fired = np.sum(state == -2)
        fractions_fired.append(n_fired / n_traps if n_traps > 0 else 0.0)

    max_len = max(len(h) for h in histories)
    padded = np.zeros((n_trials, max_len))
    for i, h in enumerate(histories):
        padded[i, :len(h)] = h
    mean_history = padded.mean(axis=0)

    return histories, mean_history, np.array(fractions_fired)

def final_history(rho, f_k, p_succ_long, p_succ_short=0.9,
                                       p_long=0.5, m=1, L=50, steps=200, rng=None):
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
