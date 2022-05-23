import numpy as np
import pandas as pd
import scipy.linalg
from numpy.linalg import matrix_power
from pyquil.noise import damping_after_dephasing
import os

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

n = 10**2
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

if not os.path.isdir(f'{t1}_{t2}_{gate_time}'):
    os.mkdir(f'{t1}_{t2}_{gate_time}')

transitions = [np.zeros((len(states), len(states)), dtype=np.half) for i in range(len(GATES))]

# building transition matrices
for ind, s in enumerate(states):
    if (ind % 500 == 0): print('.', end='')
    for i in range(100):
        state = random_state_in_reg(s)
        state_ind = states.index(dm_to_bloch_reg(state))
        for j in range(len(GATES)):
            n_state = apply_operator(state, GATES[j])
            n_state_ind = states.index(dm_to_bloch_reg(n_state))
            transitions[j][state_ind][n_state_ind] += 1

for i in range(len(GATES)):
    for j in range(len(states)):
        transitions[i][j] = np.nan_to_num(transitions[i][j] / sum(transitions[i][j]))

np.savez_compressed(f'./{t1}_{t2}_{gate_time}/transitions_{k}_{t1}_{t2}_{gate_time}.npz', transitions)
