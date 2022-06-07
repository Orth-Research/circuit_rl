import numpy as np
import pandas as pd
import scipy.linalg
from numpy.linalg import matrix_power
from pyquil.noise import damping_after_dephasing
from qiskit import *


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

def generate_target_state(n):
    s = np.matrix([1, 0])
    rho = np.outer(s, s.H)
    
    ht = GATES[0] @ GATES[1]
    rho = matrix_power(ht, n) @ rho @ matrix_power(ht.H, n)
        
    return rho

def dm_fidelity(rho, sigma):
    rho_sqrt = scipy.linalg.sqrtm(rho)
    return np.trace(scipy.linalg.sqrtm(rho_sqrt @ sigma @ rho_sqrt))**2

def trace_distance(rho, sigma):
    return np.trace(np.abs(rho - sigma)) / 2

k = 16 # discretization of the Bloch sphere
n = 10**7

GATES = [
    np.matrix([[1, 1], [1, -1]]) / np.sqrt(2), # H
    np.matrix([[1, 0], [0, np.exp(1j * np.pi / 4)]]), # T
    np.matrix([[1, 0], [0, 1]]) # I
]

thetas = np.array(pd.cut(np.linspace(0, np.pi, k), k, precision=10, include_lowest=True))
thetas[0] = pd.Interval(0, thetas[0].right, closed='both')
phis = np.array(pd.cut(np.linspace(0, 2*np.pi, 2*k), 2*k,  precision=10, include_lowest=True))
phis[0] = pd.Interval(0, phis[0].right, closed='both')
radii = pd.cut(np.linspace(0, 1, k), k, precision=10, include_lowest=True)

states = [(i, j, k) for i in range(len(thetas)) for j in range(len(phis)) for k in range(len(radii))]
values = np.zeros(len(thetas) * len(phis) * len(radii))
state_fidelities = np.zeros(len(thetas) * len(phis) * len(radii))
state_trace_distances = np.zeros(len(thetas) * len(phis) * len(radii))

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

goal = generate_target_state(n=n)
goal_reg = dm_to_bloch_reg(goal)

t1s = [1e-6, 5e-6, 10e-6, 15e-6, 20e-6, 50e-6]
t2s = [1e-6, 5e-6, 10e-6, 15e-6, 20e-6, 50e-6]
gate_time = 2e-7

def apply_operator(rho, op): # add noise
    n_rho = np.matrix([[0, 0], [0, 0]])
    for k in NOISE[3]:
        n_rho = n_rho + np.matrix(op @ k @ rho @ np.matrix(k).H @ op.H)
    # return np.matrix(op @ rho @ op.H)
    return n_rho

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

noiseless_res = []
noisy_res = []
for (t1, t2) in zip(t1s, t2s):
    p = 0.01
    NOISE = [
        [np.matrix([[1, 0], [0, 1]])],
        [(1 - p) * np.matrix([[1, 0], [0, 1]]), p/3 * np.matrix([[0, 1], [1, 0]]), p/3 * np.matrix([[0, -1j], [1j, 0]]), p/3 * np.matrix([[1, 0], [0, -1]])],
        [(1 - p) * np.matrix([[1, 0], [0, 1]]), p * np.matrix([[1, 0], [0, -1]])],
        damping_after_dephasing(t1, t2, gate_time)
    ]

    with np.load(f'./results/{t1}_{t2}_{gate_time}/transitions_{k}_{t1}_{t2}_{gate_time}.npz') as data:
        transitions = data['arr_0']
    with np.load(f'./results/{t1}_{t2}_{gate_time}/policy_{n}_{k}_{t1}_{t2}_{gate_time}.npz') as data:
        policy = data['arr_0']
    with np.load(f'./results/{t1}_{t2}_{gate_time}/v_{n}_{k}_{t1}_{t2}_{gate_time}.npz') as data:
        v = data['arr_0']
    NOISE[3] = damping_after_dephasing(t1, t2, gate_time)

    prev_seq = [0,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,0] # found from the noiseless MDP
    # prev_seq = [ 
    #     [0,1,0,1,0,1,0,1,1],
    #     [0,1,1,1,0,1,0,1,1,1],
    #     [0,1,0],
    #     [0,1,0],
    #     [0,1,1,0,1],
    #     [0,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,0],
    #     [2],
    #     [2],
    #     [0,1,0,1,0,1,0,1,1,1,0]
    # ]

    optimal_programs = []
    for i in range(k):
        converged = False
        while not converged:
            s = random_state_in_reg((0, 0, k-1))
            prog = []
            counter = 0
            while counter < 10:
                action = np.argmax(policy[states.index(dm_to_bloch_reg(s))])
                next_s = apply_operator(s, GATES[action])
                prog.append(action)
                next_s = random_state_in_reg(dm_to_bloch_reg(next_s))
                s = next_s
                counter += 1
                reg = dm_to_bloch_reg(s)
                if (reg[0] == goal_reg[0] and reg[1] == goal_reg[1]):
                    print('converged')
                    converged = True
                    break
            
        optimal_programs.append(prog)

    fidelities = []
    for prog in optimal_programs:
        psi = np.matrix([1,0])
        rho = np.matrix(np.outer(psi, psi.H))

        for g in prog:
            rho = apply_operator(rho, GATES[g])
        fidelities.append(dm_fidelity(goal, rho))

    min_ind = np.argmin(fidelities)
    noisy_res.append((optimal_programs[min_ind], fidelities[min_ind]))


    alt_progs = []
    alt_progs.append(prev_seq[int(np.log10(n))-2])


    psi = np.matrix([1,0])
    rho = np.matrix(np.outer(psi, psi.H))

    for g in prog:
        rho = apply_operator(rho, GATES[g])
    noiseless_res.append((prev_seq, dm_fidelity(goal, rho)))


print(noiseless_res)
print(noisy_res)