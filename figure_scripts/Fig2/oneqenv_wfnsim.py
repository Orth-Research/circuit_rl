import gym
import gym.spaces
import numpy as np
from pyquil import get_qc, Program
from pyquil.api import WavefunctionSimulator
from pyquil.gates import *
from collections import defaultdict


class OneQEnv(gym.Env):
    
    def __init__(self, goal_angles, gates=[H(0), T(0)], bins=(4, 8)):
        # WavefunctionSimulator
        self.wfn_sim = WavefunctionSimulator()
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
            state = np.random.choice(self.nS)
        assert state in range(self.nS), "Invalid state"
        angles = self.state_to_polar_angles(state)
        assert state == self.polar_angles_to_state(angles), f"Something went wrong in reset for state {state}"
        # consistency checks
        assert angles[0] >= 0.0 and angles[0] <= np.pi
        assert angles[1] >= 0.0 and angles[1] <= 2 * np.pi

        self.polar_angles_to_prog(angles)
        self.wfn_to_polar_angles()
        self.state = state
        return self.state
    
    
    def step(self, action, shuffle=False):
        
        if shuffle:
            return self.step_shuffle(action)
        
        else:
            return self.step_old(action)
        
        
    def step_old(self, action):
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


    def step_shuffle(self, action):

        self._program += self._actions[action]

        # first, get polar angles from the new program
        self.wfn_to_polar_angles()

        # identify the resultant state
        self.state = self.polar_angles_to_state([self._theta, self._phi])
        
        # get back the angles from this state
        theta, phi = self.state_to_polar_angles(self.state)
        assert self.state == self.polar_angles_to_state([theta, phi]), f"Something went wrong in step"
        
        # set the angles to these new values
        self._theta = theta
        self._phi = phi

        # convert the perturbed polar angles to new program
        self.polar_angles_to_prog([self._theta, self._phi])

        # convert the perturbed polar angles to new wavefunction
        self.wfn_to_polar_angles()

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
        delta_theta = np.round(self._grid[0][0], 3)
        delta_phi = np.round(self._grid[1][0], 3)
        assert state_num in range(self.nS)
        if state_num == 0:
            theta = 0.0
            phi = 0.0
#             theta = max(0.0 + np.random.uniform(0.0, delta_theta) - 0.01, 0.0)
#             phi = max(0.0 + np.random.uniform(0.0, 2 * np.pi) - 0.01, 0.0)

        elif state_num == (self._bins[0] - 2) * self._bins[1] + 1:
            theta = max(np.pi - delta_theta, np.pi - delta_theta + np.random.uniform(0.0, delta_theta) - 0.01)
            phi = max(0.0 + np.random.uniform(0.0, 2 * np.pi) - 0.01, 0.0)

        else:
            N = self._bins[0]
            M = self._bins[1]
            theta = (np.pi / N) * ((state_num - 1) // M + 1)
            theta = max(theta, theta + np.random.uniform(0.0, delta_theta) - 0.01)
            phi = (2 * np.pi / M) * ((state_num - 1) % M)
            phi = max(phi, phi + np.random.uniform(0.0, delta_phi) - 0.01)

        if theta > np.pi or theta < 0.0: theta = theta % (np.pi)
        if phi > 2 * np.pi or phi < 0.0: phi = phi % (2 * np.pi)
        # consistency checks
        assert theta >= 0.0 and theta <= np.pi, f"theta: {theta}"
        assert phi >= 0.0 and phi <= 2 * np.pi, f"phi: {phi}"
        return theta, phi
    

    def polar_angles_to_state(self, angles):
        discrete_sample = self.discretize(sample=angles, grid=self._grid)
        state_number = self.state_number(discrete_sample, mygrid=self._grid)
        return state_number
    
    
    def polar_angles_to_prog(self, angles):
        theta = angles[0]
        phi = angles[1]
        self._program = Program(RY(theta, 0), RZ(phi, 0))