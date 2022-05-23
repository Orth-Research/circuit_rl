import numpy as np
import pandas as pd

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
    theta = max(0, np.arctan(y/x))
    phi = max(0, np.arccos(z/r))

    return (theta, phi, r)

def spherical_to_cartesian(state):
    theta = state[0]
    phi = state[1]
    r = state[2]

    x = r*np.cos(theta)*np.sin(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(phi)

    return (x, y, z)

def random_state_in_reg(reg):
    # where reg is a tuple specifying (theta, phi, radius)
    # returns a density matrix
    theta = np.random.uniform(max(0, thetas[reg[0]].left), min(thetas[reg[0]].right, np.pi))

    # maybe consider the poles as one state
    # if (reg == (0, 0) or reg == (len(thetas)-1, len(phis)-1)):
        # phi = np.random.uniform(-np.pi, np.pi)
    # else:
    phi = np.random.uniform(max(-np.pi, phis[reg[1]].left), min(phis[reg[1]].right, np.pi))
    r = np.random.uniform(radii[reg[2]].left, radii[reg[2]].right)

    state = spherical_to_cartesian((theta, phi, r))
    rho = (np.eye(2) + state[0]*np.array([[0,1],[1,0]]) + state[1]*np.array([[0, -1j], [1j, 0]]) + state[2]*np.array([[1,0], [0,-1]]))/2
    return rho

def generate_target_state(n):
    s = np.matrix([1, 0])
    rho = np.outer(s, s.H)

    for i in range(n):
        ht = GATES[0] @ GATES[1]
        rho = apply_operator(rho, ht)
    return rho

def apply_operator(rho, op): # add noise
    return op @ rho @ op.conj().T

# ----------------------------------------------------------------

k = 16

GATES = [
    np.array([[1, 1], [1, -1]]) * 1/np.sqrt(2), # H
    np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]]), # T
    np.array([[1, 0], [0, 1]]) # I
]

thetas = pd.cut(np.linspace(0, np.pi, k), k, precision=10, include_lowest=True)
phis = pd.cut(np.linspace(0, 2*np.pi, k), k,  precision=10, include_lowest=True)
# radii = pd.cut(np.linspace(0, 1, k), k, precision=10, include_lowest=True)
rs = (1 - np.geomspace(1e-3, 1, k))[::-1]
rs[0] = 0
rs[-1] = 1
radii = pd.cut(rs, k, precision=10)

states = [(i, j, k) for i in range(len(thetas)) for j in range(len(phis)) for k in range(len(radii))]

transitions = [np.zeros((len(states), len(states)), dtype=np.half) for i in range(len(GATES))]

print('Building transition matrices')
# building transition matrices
for ind, s in enumerate(states):
    if (ind % 500 == 0): print('.')
    for i in range(100):
        rho = random_state_in_reg(s)
        for j, gate in enumerate(GATES):
            n_state = dm_to_bloch_reg(apply_operator(rho, gate))
            n_state_ind = states.index(n_state)
            state_ind = states.index(dm_to_bloch_reg(rho))
            transitions[j][state_ind, n_state_ind] += 1

for i in range(len(GATES)):
    for j in range(len(states)):
        transitions[i][j] = np.nan_to_num(transitions[i][j] / sum(transitions[i][j]))

np.savez_compressed('transitions_16_noiseless.npz', a=transitions)
