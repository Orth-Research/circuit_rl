import numpy as np
import pandas as pd
import scipy.linalg
from numpy.linalg import matrix_power
from pyquil.noise import damping_after_dephasing

def dm_to_bloch_reg(rho):
    # rho is a density matrix
    state = dm_to_bloch_vector(rho)
    state = cartesian_to_spherical(state)

    # state is now (theta, phi, r)
    for i, intv in enumerate(thetas):
        if (state[0] in intv):
            theta_reg = i
    for i, intv in enumerate(phis):
        if (state[1] in intv):
            phi_reg = i
    for i, intv in enumerate(radii):
        if (state[2] in intv):
            r_reg = i

    if (theta_reg == 0):
        theta_reg = phi_reg = 0
    if (theta_reg == len(thetas)-1):
        theta_reg = len(thetas)-1
        phi_reg = len(phis)-1

    return (theta_reg, phi_reg, r_reg)

def dm_to_bloch_vector(rho):
    x = np.array([[0,1],[1,0]])
    y = np.array([[0,-1j],[1j,0]])
    z = np.array([[1,0],[0,-1]])

    # where rho is a density matrix
    return (np.trace(rho @ x), np.trace(rho @ y), np.trace(rho @ z))

def dm_to_polar_coords(rho):
    # rho is a density matrix
    return cartesian_to_spherical(dm_to_bloch_vector(rho))

def cartesian_to_spherical(state):
    x = state[0]
    y = state[1]
    z = state[2]

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = max(0, np.arccos(z/r))
    phi = np.arctan2(y.real, x.real)
    if (phi < 0): phi += 2*np.pi

    return (theta, phi, r)

def spherical_to_cartesian(state):
    theta = state[0]
    phi = state[1]
    r = state[2]

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    if abs(x) < 0.0001:
        x = 0
    if abs(y) < 0.0001:
        y = 0
    if abs(z) < 0.0001:
        z = 0
    return (x, y, z)

def random_state_in_reg(reg):
    # where reg is a tuple specifying (theta, phi, radius)
    # returns a density matrix
    theta = np.random.uniform(thetas[reg[0]].left, thetas[reg[0]].right)

    # maybe consider the poles as one state
    if (reg[0] == 0 or reg == len(thetas)-1):
        phi = np.random.uniform(0, 2*np.pi)
    else:
        phi = np.random.uniform(phis[reg[1]].left, phis[reg[1]].right)
    r = np.random.uniform(radii[reg[2]].left, radii[reg[2]].right)

    state = spherical_to_cartesian((theta, phi, r))
    rho = (np.eye(2) + state[0]*np.array([[0,1],[1,0]]) + state[1]*np.array([[0, -1j], [1j, 0]]) + state[2]*np.array([[1,0], [0,-1]]))/2
    return np.matrix(rho)

def generate_target_state(n):
    s = np.matrix([1, 0])
    rho = np.outer(s, s.H)

    ht = GATES[0] @ GATES[1]
    rho = matrix_power(ht, n) @ rho @ matrix_power(ht.H, n)

    return rho

def apply_operator(rho, op): # add noise
    n_rho = np.matrix([[0, 0], [0, 0]])
    for k in NOISE[3]:
        n_rho = n_rho + np.matrix(op @ k @ rho @ np.matrix(k).H @ op.H)
    # return np.matrix(op @ rho @ op.H)
    return n_rho

def dm_fidelity(rho, sigma):
    rho_sqrt = scipy.linalg.sqrtm(rho)
    return np.trace(scipy.linalg.sqrtm(rho_sqrt @ sigma @ rho_sqrt))**2

n = 10**7
k = 16

GATES = [
    np.matrix([[1, 1], [1, -1]]) / np.sqrt(2), # H
    np.matrix([[1, 0], [0, np.exp(1j * np.pi / 4)]]), # T
    np.matrix([[1, 0], [0, 1]]) # I
]

p = 0.01
t1 = 75e-6
t2 = 75e-6
gate_time = 2e-7

NOISE = [
    [np.matrix([[1, 0], [0, 1]])],
    [(1 - p) * np.matrix([[1, 0], [0, 1]]), p/3 * np.matrix([[0, 1], [1, 0]]), p/3 * np.matrix([[0, -1j], [1j, 0]]), p/3 * np.matrix([[1, 0], [0, -1]])],
    [(1 - p) * np.matrix([[1, 0], [0, 1]]), p * np.matrix([[1, 0], [0, -1]])],
    damping_after_dephasing(t1, t2, gate_time)
]

goal = generate_target_state(n=n)
thetas = np.array(pd.cut(np.linspace(0, np.pi, k), k, precision=10, include_lowest=True))
thetas[0] = pd.Interval(0, thetas[0].right, closed='both')
phis = np.array(pd.cut(np.linspace(0, 2*np.pi, 2*k), 2*k,  precision=10, include_lowest=True))
phis[0] = pd.Interval(0, phis[0].right, closed='both')
radii = pd.cut(np.linspace(0, 1, k), k, precision=10, include_lowest=True)

goal_reg = dm_to_bloch_reg(goal)

states = [(i, j, k) for i in range(len(thetas)) for j in range(len(phis)) for k in range(len(radii))]
values = np.zeros(len(thetas) * len(phis) * len(radii))

with np.load(f'./{t1}_{t2}_{gate_time}/transitions_{k}_{t1}_{t2}_{gate_time}.npz') as data:
    transitions = data['arr_0']

def R(state, action):
    if (state[0] == goal_reg[0] and state[1] == goal_reg[1]):
        return state[2]/k # pretty much the purity of the state
    else:
        return 0

def policy_eval(policy, discount_factor=0.8, epsilon=0.001):
    V_old = np.zeros(len(states))
    while True:
    # for i in range(1):
        V_new = np.zeros(len(states))
        delta = 0
        for s, _ in enumerate(states):
            v_fn = 0
            action_probs = policy[s]
            for a, _ in enumerate(GATES):
                p_trans = transitions[a][s]
                p_next_states = np.nonzero(transitions[a][s])[0]
                for next_s in p_next_states:
                    v_fn += action_probs[a] * p_trans[next_s] * (R(states[s], a) + discount_factor * V_old[next_s])
            delta = max(delta, abs(v_fn - V_old[s]))
            V_new[s] = v_fn
        V_old = V_new
        if(delta < epsilon):
            print('converged')
            break
    # since technically the entire north/south pole is one state, copy (0, 0) and (k-1, k-1) over
    # won't ever be used, but it is needed for the visualization

    for i in range(k):
        ind1 = states.index((0,0,i))
        ind2 = states.index((k-1, k-1, i))
        for j in range(1, len(phis)):
            V_old[ind1 + j*k] = V_old[ind1]
            V_old[ind2 - j*k] = V_old[ind2]
    return np.array(V_old)

def policy_improvement(policy_eval_fn=policy_eval, discount_factor=0.8):
    def one_step_lookahead(s, V_old):
        actions = np.zeros(len(GATES))
        for a in range(len(GATES)):
            v_fn = 0
            p_trans = transitions[a][s]
            p_next_states = np.nonzero(transitions[a][s])[0]
            for next_s in p_next_states:
                v_fn += p_trans[next_s] * (R(states[s], a) + discount_factor * V_old[next_s])
            actions[a] = v_fn
        return actions
    policy = np.ones([len(states), len(GATES)]) / len(GATES)
    actions_values = np.zeros(len(GATES))

    while True:
        value_fn = policy_eval_fn(policy)
        policy_stable = True
        for s in range(len(states)):
            actions_values = one_step_lookahead(s, value_fn)
            best_action = np.argmax(actions_values)
            chosen_action = np.argmax(policy[s])
            if(best_action != chosen_action):
                policy_stable = False
            policy[s] = np.eye(len(GATES))[best_action]

        if(policy_stable):
            return policy, value_fn

policy, v = policy_improvement(policy_eval)
np.savez_compressed(f'./{t1}_{t2}_{gate_time}/policy_{n}_{k}_{t1}_{t2}_{gate_time}.npz', policy)
np.savez_compressed(f'./{t1}_{t2}_{gate_time}/v_{n}_{k}_{t1}_{t2}_{gate_time}.npz', v)
