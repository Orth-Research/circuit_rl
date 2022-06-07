import gym
import gym.spaces
import numpy as np
from pyquil import get_qc, Program
from pyquil.api import WavefunctionSimulator
from pyquil.gates import *
import sys
from pyquil.api._base_connection import ForestConnection
import pickle

from collections import defaultdict
import sys


# identify discrete gates on qubit 0
num_angles = 1000
angles = np.linspace(0.0, 2 * np.pi, num_angles)
gates = [RY(theta, 0) for theta in angles]
gates += [RZ(theta, 0) for theta in angles]

pickle.dump(np.array([1, 2, 3]), open('trial_dump.p', 'wb'))


class OneQEnv(gym.Env):
    
    def __init__(self, goal_angles, bins=(4, 8)):
        # WavefunctionSimulator
        self.wfn_sim = WavefunctionSimulator(connection=ForestConnection(sync_endpoint='http://localhost:5001'))
        # Identify discrete state space grid
        self._bins = bins
        self._grid = self.create_uniform_grid(bins=self._bins)
        # Identify desired state
        self.goal_angles = goal_angles
        self.goal_state = self.polar_angles_to_state(self.goal_angles)
        # Create rewards dictionary
        self.d_rewards = defaultdict(lambda: 0)
        self.d_rewards[self.goal_state] = 1
        # Identify discrete state space
        self.state_space = gym.spaces.Discrete((self._bins[0] - 2) * self._bins[1] + 2)
        self.observation_space = self.state_space
        # Identify discrete action space
        self._actions = gates
        self.action_space = gym.spaces.Discrete(len(self._actions))
        # Identify sizes of state and action spaces
        self.nS = self.state_space.n
        self.nA = self.action_space.n
        # Initialize to identity program
        self._program = Program(I(0))
        # Identify state
        self.wfn_to_polar_angles()
        self.state = self.polar_angles_to_state([self._theta, self._phi])
        # consistency checks for polar angles
        assert self._theta >= 0.0 and self._theta <= np.pi
        assert self._phi >= 0.0 and self._phi <= 2 * np.pi
        self.info = {}
        
        
    def reset(self, state=None):
        if state is None:
            state = np.random.choice(env.nS)
        assert state in range(env.nS), "Invalid state"
        angles = self.state_to_polar_angles(state)
        # consistency checks
        assert angles[0] >= 0.0 and angles[0] <= np.pi
        assert angles[1] >= 0.0 and angles[1] <= 2 * np.pi

        self.polar_angles_to_prog(angles)
        self.state = state
        return self.state


    def step(self, action):
        self._program += self._actions[action]
        self.wfn_to_polar_angles()
        self.state = self.polar_angles_to_state([self._theta, self._phi])
        # consistency checks
        assert self._theta >= 0.0 and self._theta <= np.pi, f"theta: {self._theta} in step; wavefunction: {self._wfn}; wfn_polar: {self._wfn_polar}"
        assert self._phi >= 0.0 and self._phi <= 2 * np.pi, f"phi: {self._phi} in step; wavefunction: {self._wfn}; wfn_polar: {self._wfn_polar}"
        if self.state == self.goal_state:
            reward = 1
            done = True
        else:
            reward = 0
            done = False

        return self.state, reward, done, self.info


    @classmethod
    def amps_to_polar_angles(self, z):
        # NOTE: This is a poorly named function
        a = z.real
        b = z.imag
        r = np.sqrt(a**2 + b**2)
        angle = np.arctan2(b, a)
        return r, angle
    
    
    @classmethod
    def amps_to_actual_polar_angles(self, z):
        a = z.real
        b = z.imag
        r = np.sqrt(a**2 + b**2)
        angle = np.arctan2(b, a)
        cos_theta_over_2 = np.abs(r[0])
        if cos_theta_over_2 > 1.0: cos_theta_over_2 = 1.0
        theta = 2 * np.arccos(cos_theta_over_2)
        theta = theta % (np.pi)
        phi = angle[1] - angle[0]
        if phi < 0.0: phi += 2 * np.pi
        phi = phi % (2 * np.pi)
        return theta, phi


    def wfn_to_polar_angles(self):
        self._wfn = self.wfn_sim.wavefunction(self._program)
        self._wfn_amps = self._wfn.amplitudes
        self._wfn_polar = self.amps_to_polar_angles(self._wfn_amps)
        cos_theta_over_2 = np.abs(self._wfn_polar[0][0])
        if cos_theta_over_2 > 1.0: cos_theta_over_2 = 1.0
        self._theta = 2 * np.arccos(cos_theta_over_2)
        if self._theta < 0.0: self._theta = -self._theta
        self._phi = self._wfn_polar[1][1] - self._wfn_polar[1][0]
        if self._phi < 0.0: self._phi += 2 * np.pi
        # consistency checks
        assert self._theta >= 0.0 and self._theta <= np.pi, f"theta: {self._theta} in wfn_to_polar_angles; wavefunction: {self._wfn}; wfn_polar: {self._wfn_polar}"
        assert self._phi >= 0.0 and self._phi <= 2 * np.pi, f"phi: {self._phi} in wfn_to_polar_angles; wavefunction: {self._wfn}; wfn_polar: {self._wfn_polar}"


    def polar_angles_to_state(self, angles):
        discrete_sample = self.discretize(sample=angles, grid=self._grid)
        state_number = self.state_number(discrete_sample, mygrid=self._grid)
        return state_number


    def create_uniform_grid(self, bins):
        # create grid for angles
        low=[0.0, 0.0]
        high=[np.pi, 2 * np.pi]
        grid_list = []
        for i in range(len(low)):
            tmp_high = high[i]
            tmp_low = low[i]
            tmp_bin = bins[i]
            tmp_diff = (tmp_high - tmp_low) / tmp_bin
            tmp_arr = np.linspace(tmp_low + tmp_diff, tmp_high - tmp_diff, tmp_bin - 1)
            grid_list.append(tmp_arr)
        return grid_list
    
    
    def discretize(self, sample, grid):
        # Discretize a sample as per given grid.
        samps = []
        dims = len(grid)
        for i in range(dims):
            tmp_samp = np.digitize(np.round(sample[i], 2), np.round(grid[i], 2))
            samps.append(tmp_samp)

        samps = np.array(samps)
        assert samps.shape == (dims, )
        return samps


    def state_number(self, discrete_sample, mygrid):
        # expecting an array with 2 entries
        if discrete_sample[0] == 0:
            return 0
        elif discrete_sample[0] == self._bins[0] - 1:
            return (self._bins[0] - 2) * self._bins[1] + 1
        else:
            return (discrete_sample[0] - 1) * self._bins[1] + discrete_sample[1] + 1


    def state_to_polar_angles(self, state_num):
        assert state_num in range(self.nS)
        if state_num == 0:
            return 0.0, 0.0
        elif state_num == (self._bins[0] - 2) * self._bins[1] + 1:
            return np.pi, 0.0
        else:
            N = self._bins[0]
            M = self._bins[1]
            theta = (np.pi / N) * ((state_num - 1) // M + 1)
            theta = theta % (2 * np.pi)
            phi = (2 * np.pi / M) * ((state_num - 1) % M)
            phi = phi % (2 * np.pi)
            # consistency checks
            assert theta >= 0.0 and theta <= np.pi, f"theta: {theta}"
            assert phi >= 0.0 and phi <= 2 * np.pi, f"phi: {phi}"
            return theta, phi
    
    
    def polar_angles_to_prog(self, angles):
        theta = angles[0]
        phi = angles[1]
        self._program = Program(RY(theta, 0), RZ(phi, 0))



def policy_evaluation_v0(env, policy, d_raw_counts, gamma=0.8, theta=1e-8):
    # Note: this is an old version of policy_evaluation that has a bug.
    # TODO: change the function name `policy_evalution` to `policy_evaluation_v0`
    #       in all RL_discrete_1q_space_vXXX for XXX in [0..14] to recover same results,
    #       or produce new results using the updated function
    V = np.zeros(env.nS)
    converged = False
    counter = 0
    while not converged:
        counter += 1
        print(f"\rCounter for policy evaluation: {counter}", end="")
        sys.stdout.flush()
        delta = 0
        for s in range(env.nS):
            vold = V[s]
            vnew = 0
            for a in range(env.nA):
                _ = env.reset(s)
                next_state, reward, done, info = env.step(a)
                prob_next = d_raw_counts[s, a].count(next_state) / len(d_raw_counts[s, a])
                vnew += policy[s][a] * prob_next * (reward + gamma * V[next_state])
            V[s] = vnew
            delta = max(delta, np.abs(vold - V[s]))
        if delta < theta:
            converged = True
            
    return 


def policy_evaluation(env, policy, d_raw_counts, gamma=0.8, theta=1e-8):
    # Note: this newer version requires that the `env` contain a dict
    # containing rewards as a function of state as well
    V = np.zeros(env.nS)
    converged = False
    counter = 0
    while not converged:
        counter += 1
        print(f"\rCounter for policy evaluation: {counter}", end="")
        sys.stdout.flush()
        delta = 0
        for s in range(env.nS):
            vold = V[s]
            vnew = 0
            for a in range(env.nA):
                for next_state in set(d_raw_counts[s, a]):
                    prob_next = d_raw_counts[s, a].count(next_state) / len(d_raw_counts[s, a])
                    vnew += policy[s][a] * prob_next * (env.d_rewards[next_state] + gamma * V[next_state])
            V[s] = vnew
            delta = max(delta, np.abs(vold - V[s]))
        if delta < theta:
            converged = True
            
    return V


def policy_improvement_v0(env, V, d_raw_counts, gamma=0.8):
    # Note: this is an old version of policy_improvement that has a bug.
    # TODO: change the function name `policy_improvement` to `policy_improvement_v0`
    #       in all RL_discrete_1q_space_vXXX for XXX in [0..14] to recover same results,
    #       or produce new results using the updated function

    # this function should improve on the policy that produced value function V
    Q = np.zeros(shape=(env.nS, env.nA))
    policy = np.zeros(shape=(env.nS, env.nA))
    counter_improve = 0
    for s in range(env.nS):
        for a in range(env.nA):
            counter_improve += 1
            print(f"\rCounter for policy improvement: {counter_improve}", end="")
            sys.stdout.flush()
            _ = env.reset(s)
            next_state, reward, done, info = env.step(a)
            prob_next = d_raw_counts[s, a].count(next_state) / len(d_raw_counts[s, a])
            Q[s, a] = prob_next * (reward + gamma * V[next_state])
        best_a = np.argmax(Q[s])
        policy[s][best_a] = 1
    
    return policy


def policy_improvement(env, V, d_raw_counts, gamma=0.8):
    # This follows the pseudo-code on pg. 65 of Sutton and Barto
    # this function should improve on the policy that produced value function V
    policy = np.zeros(shape=(env.nS, env.nA))
    for s in range(env.nS):
        possible_policies = []
        for a in range(env.nA):
            v = 0
            for next_state in set(d_raw_counts[s, a]):
                prob_next = d_raw_counts[s, a].count(next_state) / len(d_raw_counts[s, a])
                v += prob_next * (env.d_rewards[next_state] + gamma * V[next_state])
            possible_policies.append(v)
        best_a = np.argmax(possible_policies)
        policy[s][best_a] = 1

    return policy


def policy_iteration(env, d_raw_counts, theta=1e-8):
    policy = np.ones((env.nS, env.nA)) / env.nA
    counter_iter = 0
    while True:
        counter_iter += 1
        print(f"\rCounter for policy iteration: {counter_iter}", end="")
        sys.stdout.flush()
        V = policy_evaluation(env, policy, d_raw_counts, theta=theta)
        improved_policy = policy_improvement(env, V, d_raw_counts)
        if np.max(np.abs(policy - improved_policy)) < theta:
            break
        policy = improved_policy
    return policy


def policy_iteration_v2(env, d_raw_counts, theta=1e-8):
    # NOTE: This calls the new implementions of the functions `policy_evaluation` and
    # `policy_improvement`. The notebooks RL_discrete_1q_space_vXXX for XXX in [0..14]
    # all use the older implementations of those functions, so to recover those (perhaps false)
    # results, we'd need to implement this function using the renamed functions ending with `_v0`
    policy = np.ones((env.nS, env.nA)) / env.nA
    counter_iter = 0
    while True:
        counter_iter += 1
        print(f"\rCounter for policy iteration: {counter_iter}", end="")
        sys.stdout.flush()
        V = policy_evaluation(env, policy, d_raw_counts, theta=theta)
        improved_policy = policy_improvement(env, V, d_raw_counts)
        if np.max(np.abs(policy - improved_policy)) < theta:
            break
        policy = improved_policy
    return policy, counter_iter


env = OneQEnv(goal_angles=np.array([np.pi, 0.0]), bins=(16, 32))
print(f"No. of states: {env.nS}")


# draw angles uniformly over the Bloch sphere


size_expt = 10000
d_raw_counts = defaultdict(lambda: [])


for expt in range(size_expt):
    print(f"\rExperiment {expt+1} / {size_expt}", end="")
    sys.stdout.flush()
    u = np.random.uniform(0.0, 1.0, size=2)
    theta = np.arccos(2 * u[0] - 1)
    phi = 2 * np.pi * u[1]
    angles = [theta, phi]
    
    for a in range(env.nA):
        env.reset()
        env.polar_angles_to_prog(angles)   # sets env._program
        s = env.polar_angles_to_state(angles)    # sets nothing; returns state number
        next_s, reward, done, info = env.step(a)
        d_raw_counts[s, a] += [next_s]


# pickle.dump(d_raw_counts, open('d_raw_counts_v11.p', 'wb'))


optimal_policy, counter_iter = policy_iteration_v2(env, d_raw_counts)
print("\n")
print(f"Total number of policy iterations: {counter_iter}")

pickle.dump(optimal_policy, open('optimal_policy_discrete_1q_v20.p', 'wb'))

V_optimal = policy_evaluation(env, optimal_policy, d_raw_counts)
pickle.dump(V_optimal, open('V_optimal_discrete_1q_v20.p', 'wb'))

