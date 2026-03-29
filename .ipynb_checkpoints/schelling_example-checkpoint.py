 #!/usr/bin/env python -W ignore::DeprecationWarning

from environment import Environment

from itertools import count
from multiprocessing import Process, Lock

import time
import random
import os, sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import numpy as np

class Schelling(Environment):
    def __init__(self, size, p_hunter = 0.05, p_prey = 0,
            prey_reward = 1, stuck_penalty = 1,
            death_penalty = 1, p_resurrection = 0.2,
            agent_max_age = 100, agent_range = 2, num_actions = 5,
            same = True, lock = None,
            max_iteration = 5000, name = None, eating_bonus = 1,
            alpha=1., beta=1., gamma = 1.):

        super(Schelling, self).__init__(size, p_hunter, p_prey,
                prey_reward = prey_reward, stuck_penalty = stuck_penalty,
                death_penalty = death_penalty, p_resurrection = p_resurrection,
                agent_max_age = agent_max_age, agent_range = agent_range, num_actions = num_actions,
                same = same, lock = lock, name = name, max_iteration = max_iteration)

        self.eating_bonus = prey_reward

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.alive_reward = 0.1

    def default(self, agent):
        curr = self.get_agent_state(agent)
        prev = agent.get_state()
        default = (agent.get_type() * (curr - prev))
        sames = default[default > 0.1].sum()
        diffs = self.alpha * default[default < -0.1].sum()
        return sames + diffs

    def on_free(self, agent):
        self.move(agent)
        return self.beta * self.alive_reward + self.default(agent)

    def on_opponent(self, agent, opponent):
        _ = self.kill(opponent, killer=agent)
        return self.beta * self.alive_reward + self.default(agent) + self.prey_reward * self.gamma

    def on_still(self, agent):
        return -10*self.alive_reward

    def on_obstacle(self, agent):
        return -10*self.alive_reward
    def on_same(self, agent, other):
        return -10*self.alive_reward

    def kill(self, victim, killer=False):
        if victim.get_type() in [-1, 1]:
            id = victim.get_id()
            if id in self.id_to_type:
                self.id_to_lives[id].append(victim.get_age())
            else:
                self.id_to_type[id] = victim.get_type()
                self.id_to_lives[id] = [victim.get_age()]
        i, j = victim.get_loc()

        self.map[i, j] = 0
        state = self.get_agent_state(victim)
        del self.loc_to_agent[(i, j)]

        victim.die(state, -self.death_penalty)
        if killer:
            killer.eat(self.gamma * 1)
            self.move(killer)
        return -self.death_penalty

# ── Paper grid convention: A=-1, free=0, B=+1, prey=2 ────────────────────────

def segregation_index(grid, agent_type, radius=1):
    """
    Fraction of non-empty, non-prey neighbors that share `agent_type`,
    averaged over all agents of that type.
    Toroidal boundary conditions.
    """
    N, M = grid.shape
    scores = []
    for r in range(N):
        for c in range(M):
            if grid[r, c] != agent_type:
                continue
            same = total = 0
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if dr == 0 and dc == 0:
                        continue
                    val = grid[(r + dr) % N, (c + dc) % M]
                    if val in (-1, 1):          # only count A and B
                        total += 1
                        if val == agent_type:
                            same += 1
            if total > 0:
                scores.append(same / total)
    return float(np.mean(scores)) if scores else 0.0

def global_segregation(grid, radius=1):
    """Mean of type-A and type-B segregation indices."""
    sa = segregation_index(grid, -1, radius)
    sb = segregation_index(grid,  1, radius)
    return (sa + sb) / 2

def cluster_sizes(grid, agent_type):
    """Sorted list of connected-component sizes for one agent type."""
    mask = (grid == agent_type).astype(int)
    labeled, n = scipy_label(mask)
    sizes = [int(np.sum(labeled == k)) for k in range(1, n + 1)]
    return sorted(sizes, reverse=True)

def count_agents(grid):
    """Return dict with counts of A, B, prey, empty."""
    return {
        'A':     int(np.sum(grid == -1)),
        'B':     int(np.sum(grid ==  1)),
        'prey':  int(np.sum(grid ==  2)),
        'empty': int(np.sum(grid ==  0)),
    }

print('Metrics defined.')

def run_instrumented(society, iterations, snap_steps=None, eps=1e-6, verbose=True):
    """
    Instrumented version of the authors' play() loop.

    Returns
    -------
    metrics   : dict with per-step lists of seg_A, seg_B, seg_global,
                reward_A, reward_B, n_A, n_B
    snapshots : dict {step: grid_array}
    """
    from itertools import count as icount

    agents     = society.get_agents()
    directions = [(-1, 0), (0, -1), (1, 0), (0, 1), (0, 0)]  # paper order

    if snap_steps is None:
        snap_steps = set()
    else:
        snap_steps = set(snap_steps)

    metrics = {
        'seg_A': [], 'seg_B': [], 'seg_global': [],
        'reward_A': [], 'reward_B': [],
        'n_A': [], 'n_B': [],
        'converged_at': None,
    }
    snapshots = {}
    c = 0          # consecutive convergence counter
    times = 0

    state = society.get_map()

    for t in icount():
        t_start = time.time()
        random.shuffle(agents)

        rews   = {'A': 0.0, 'B': 0.0, 'prey': 0.0}
        counts_ = {'A': 0,   'B': 0,   'prey': 0}

        for agent in agents:
            towards = None
            name = society.vals_to_names[agent.get_type()]
            if agent.is_alive():
                agent_state = agent.get_state()
                action      = agent.decide(agent_state)   # original Agent.decide()
                towards     = directions[action]
            rew = society.step(agent, towards)            # original Environment.step()
            rews[name]    += rew
            counts_[name] += 1

        society.update()                                  # trains A_mind / B_mind
        society.record(rews)

        next_state = society.get_map()

        # ── Record metrics ───────────────────────────────────────────────────
        c_A  = counts_['A'] or 1
        c_B  = counts_['B'] or 1
        sg   = global_segregation(next_state, radius=1)
        sa   = segregation_index(next_state, -1, radius=1)
        sb   = segregation_index(next_state,  1, radius=1)
        cnt  = count_agents(next_state)

        metrics['seg_A'].append(sa)
        metrics['seg_B'].append(sb)
        metrics['seg_global'].append(sg)
        metrics['reward_A'].append(rews['A'] / c_A)
        metrics['reward_B'].append(rews['B'] / c_B)
        metrics['n_A'].append(cnt['A'])
        metrics['n_B'].append(cnt['B'])

        if t in snap_steps:
            snapshots[t] = next_state.copy()

        # ── Convergence check (paper's original logic) ───────────────────────
        if np.abs(next_state - state).sum() < eps:
            c += 1
        else:
            c = 0

        # ── Progress ─────────────────────────────────────────────────────────
        times += time.time() - t_start
        if verbose and (t % 50 == 0 or t == iterations - 1):
            print(f'  step {t:4d}/{iterations} | '
                  f'Seg={sg:.3f} (A={sa:.3f} B={sb:.3f}) | '
                  f'R_A={rews["A"]/c_A:+.3f}  R_B={rews["B"]/c_B:+.3f} | '
                  f'N_A={cnt["A"]} N_B={cnt["B"]} | '
                  f'avg_t={times/(t+1):.2f}s')

        if t == iterations - 1 or c == 20:
            if c == 20:
                metrics['converged_at'] = t
                print(f'\n  *** Converged at step {t} (20 consecutive unchanged maps) ***')
            break

        state = next_state

    society.save(0)   # write crystal.npy.gz  (original save method)
    print('\nSimulation finished.')
    return metrics, snapshots

##Ajouts pour le calcul des valeurs moyennes sur contexte de Q

ACTION_NAMES = ['Up', 'Left', 'Down', 'Right', 'Stay']
ACTION_DIRS = [(-1,0),(0,-1),(1,0),(0,1),(0,0)]
N_ACTIONS = 5

R = 3 ##à changer selon taille du contexte des agents
PATCH_W = 2*R + 1
N_CELLS = PATCH_W**2
N_NEIGH = N_CELLS-1
CX = R
CY = R

SAME_VAL  = {'A': -1, 'B':  1}
OPP_VAL   = {'A':  1, 'B': -1}

SEED = 123
def _rng(rng):
    return rng or np.random.default_rng(SEED) #on fixe la seed pour reproducibilité + equivalence statistique entre agents A et B

def _counts(f_same, f_opp, n):
    n_same = int(np.clip(round(f_same * n), 0, n))
    n_opp  = int(np.clip(round(f_opp  * n), 0, n - n_same))
    return n_same, n_opp

def _assign(patch, coords, n_same, n_opp, sv, ov, rng):
    idx = rng.choice(len(coords), n_same + n_opp, replace=False)
    for i, k in enumerate(idx):
        r, c = coords[k]
        patch[r, c] = sv if i < n_same else ov

def make_patch(agent_type, f_same, f_opp, rng=None, n_avg=1):
    rng = _rng(rng)
    sv, ov = SAME_VAL[agent_type], OPP_VAL[agent_type]

    def one():
        p = np.zeros((PATCH_W, PATCH_W), np.float32)
        p[CX, CY] = sv

        coords = [(r, c) for r in range(PATCH_W) for c in range(PATCH_W)
                  if (r, c) != (CX, CY)]

        n_same, n_opp = _counts(f_same, f_opp, N_NEIGH)
        _assign(p, coords, n_same, n_opp, sv, ov, rng)
        return p

    if n_avg == 1:
        return one()

    return np.mean([one() for _ in range(n_avg)], axis=0).astype(np.float32)

def make_patch_realistic(agent_type, f_same, f_opp, rng=None, empty_cross=True):
    rng = _rng(rng)
    sv, ov = SAME_VAL[agent_type], OPP_VAL[agent_type]

    p = np.zeros((PATCH_W, PATCH_W), np.float32)
    p[CX, CY] = sv

    cross = {(CX-1, CY), (CX, CY-1), (CX+1, CY), (CX, CY+1)}
    blocked = {(CX, CY)} | (cross if empty_cross else set())

    coords = [(r, c) for r in range(PATCH_W) for c in range(PATCH_W)
              if (r, c) not in blocked]

    n_same, n_opp = _counts(f_same, f_opp, len(coords))
    _assign(p, coords, n_same, n_opp, sv, ov, rng)

    return p

def patch_to_tensor(patch):
    return torch.tensor(patch, dtype=torch.float32)[None, None]

def get_qvals(mind, patch, age=50.0):
    x = patch_to_tensor(patch)
    age = torch.tensor([[age]], dtype=torch.float32)
    with torch.no_grad():
        return mind.network(x, age).squeeze().numpy()

def get_qvals_avg(mind, agent_type, f_same, f_opp, type_sign,
                  age=50.0, n_avg=100, rng=None):

    rng = _rng(rng)
    acc = np.zeros(N_ACTIONS)

    for _ in range(n_avg):
        p = make_patch_realistic(agent_type, f_same, f_opp, rng)
        x = patch_to_tensor(type_sign * p)
        age_t = torch.tensor([[age]], dtype=torch.float32)

        with torch.no_grad():
            acc += mind.network(x, age_t).squeeze().numpy()

    return acc / n_avg

def greedy_action(mind, patch, age=50.0):
    return int(np.argmax(get_qvals(mind, patch, age)))

def play(map, episodes, iterations, eps=1e-6, log_q=False, log_every=10, n_avg=200):
    # map.configure(prey_reward, stuck_penalty, agent_max_age)
    agents = map.get_agents()
    directions = [(-1, 0), (0, -1), (1, 0), (0, 1), (0, 0)]
    times = 0
    
    if log_q:
        q_A_time = []
        q_B_time = []
        
    for episode in range(episodes):
        c = 0
        for t in count():
            
            t_start = time.time()
            state = map.get_map()
            random.shuffle(agents)

            keys = ["A", "B", "prey"]
            rews = {key: 0 for key in keys}
            counts = {key: 0 for key in keys}
            for agent in agents:
                towards = None
                name = map.vals_to_names[agent.get_type()]
                if agent.is_alive():
                    agent_state = agent.get_state()
                    action = agent.decide(agent_state)
                    towards = directions[action]
                rew = map.step(agent, towards)
                rews[name] += rew
                counts[name] += 1

            map.update()

            if log_q and (t % log_every == 0):
                q_A = get_qvals_avg(
                    map.A_mind, 'A',
                    f_same=0.042, f_opp=0.05,
                    type_sign=-1,
                    n_avg=n_avg
                )

                q_B = get_qvals_avg(
                    map.B_mind, 'B',
                    f_same=0.0425, f_opp=0.05,
                    type_sign=1,
                    n_avg=n_avg
                )

                q_A_time.append(q_A)
                q_B_time.append(q_B)

            map.record(rews)

            next_state = map.get_map()

            time_elapsed = time.time() - t_start
            times += time_elapsed
            avg_time = times / (t + 1)
            print("I: %d\tTime Elapsed: %.2f" % (t+1, avg_time), end='\r')
            if abs(next_state - state).sum() < eps:
                c += 1

            if t == (iterations - 1) or c == 20:
                break

            state = next_state
        map.save(episode)
    print("SIMULATION IS FINISHED.")
    if log_q:
        return np.array(q_A_time), np.array(q_B_time)


def replay_snapshots(society, iterations=250, snap_steps=None, eps=1e-6, verbose=True):
    """
    Replay a trained society in pure exploitation mode (ε=0, no training)
    and return grid snapshots at requested steps.

    Parameters
    ----------
    society    : trained Schelling object
    iterations : max steps to run
    snap_steps : list of steps to snapshot; None → [0, 25%, 50%, 75%, 100%]
    eps        : convergence threshold (same as play)
    verbose    : print progress

    Returns
    -------
    snapshots  : dict {step: (H, W) numpy array}
    """
    import time
    from itertools import count as icount

    agents     = society.get_agents()
    directions = [(-1,0),(0,-1),(1,0),(0,1),(0,0)]

    if snap_steps is None:
        snap_steps = set([0, iterations//4, iterations//2, 3*iterations//4, iterations-1])
    else:
        snap_steps = set(snap_steps)

    # ── Force ε=0 on both minds so decisions are purely greedy ───────────────
    original_eps = {}
    for name, mind in [('A', society.A_mind), ('B', society.B_mind)]:
        original_eps[name] = mind.steps_done
        mind.steps_done = int(1e9)   # drives ε → EPS_END = 0 via the decay formula

    # ── Eval mode on the inner networks ──────────────────────────────────────
    society.A_mind.network.eval()
    society.B_mind.network.eval()

    snapshots = {}
    state     = society.get_map()
    c         = 0

    if 0 in snap_steps:
        snapshots[0] = state.copy()

    for t in icount(1):
        t0 = time.time()
        random.shuffle(agents)

        for agent in agents:
            towards = None
            if agent.is_alive():
                action  = agent.decide(agent.get_state())
                towards = directions[action]
            society.step(agent, towards)

        # ── No society.update() → no training ────────────────────────────────
        next_state = society.get_map()

        if t in snap_steps:
            snapshots[t] = next_state.copy()

        if verbose and t % 50 == 0:
            print(f'  step {t:4d}/{iterations}  |  elapsed {time.time()-t0:.2f}s', end='\r')

        if abs(next_state - state).sum() < eps:
            c += 1
        else:
            c = 0

        if t == iterations - 1 or c == 20:
            # Always capture the final frame
            snapshots[t] = next_state.copy()
            if verbose:
                reason = f'converged at step {t}' if c == 20 else f'reached max iterations ({iterations})'
                print(f'\n  Stopped: {reason}')
            break

        state = next_state

    # ── Restore original steps_done so society is unaffected ─────────────────
    society.A_mind.steps_done = original_eps['A']
    society.B_mind.steps_done = original_eps['B']
    society.A_mind.network.train()
    society.B_mind.network.train()

    return snapshots


if __name__ == '__main__':
    [_, name, iterations, agent_range, prey_reward, max_age, alpha, beta, gamma] = sys.argv

    # alpha is for schelling reward
    # beta is for vigilance reward
    # gamma is for interdependence reward

    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)

    episodes = 1
    iterations = int(iterations)
    l = Lock()

    args = ["Name",
            "Prey Reward",
            "Stuck Penalty",
            "Death Penalty",
            "Agent Max Age",
            "Agent Field of View"]

    society = Schelling

    play(society((50, 50), agent_range = int(agent_range),
        prey_reward = int(prey_reward), name=name,
        agent_max_age = int(max_age), max_iteration = int(iterations),
        lock=l, alpha=float(alpha), beta=float(beta), gamma=float(gamma)), 1, iterations)
