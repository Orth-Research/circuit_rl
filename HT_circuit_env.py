import numpy as np
from gym.utils import seeding
from gym.spaces import Discrete, Tuple, Box
import gym
from qiskit.quantum_info import state_fidelity
from qiskit import *
from numpy.linalg import matrix_power

GATES = {
    0: np.array([[1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)]]),
    1: np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]]),
    2: np.array([[1, 0], [0, 1]])
}

def generate_target_circuit(n=1):
    """Generates a random supremacy-style circuit using a specific set of gates.
    Args:
        n: number of cycles to generate
    Returns:
        s: 
    """
    
    s = np.array([1, 0])
    ht = GATES[0] @ GATES[1]

    s = matrix_power(ht, n) @ s

    return s

def statevector_to_bloch_reg(state, k):
    """returns
    """
    thetas = np.linspace(0, np.pi, k)
    phis = np.linspace(0, 2*np.pi, k)
    theta = 2 * np.arccos(abs(state[0]))
    phi = np.angle(state[1])

    # take into consideration the poles
    reg = (np.abs(thetas - theta).argmin(), np.abs(phis - phi).argmin())
    if (reg[0] == 0):
        reg = (0, 0)
    return reg

class CircuitEnv(gym.Env):
    """
    The main API methods that users of this class need to know are:
        step
        reset
        render
        close
        seed
    And set the following attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    """

    metadata = {'render.modes': ['human', 'rgb_array']}
    reward_range = (-1, 1)

    def __init__(self, n=1, k=32):
        self.goal = generate_target_circuit(n=n)
        # self.goal_cir = QuantumCircuit.from_qasm_str(qc)

        self.n = n
        self.k = k
        # self.bend = Aer.get_backend('statevector_simulator')
        # self.cir = f"OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];"
        self.cir = np.array([1, 0])

        # self.goal = execute(self.goal_cir, self.bend).result().get_statevector(self.goal_cir)
        self.goal_reg = statevector_to_bloch_reg(self.goal, self.k)
        self.gate_counts = [0, 0, 0]
        # Three actions
        #   1. H gate
        #   2. T gate
        #   3. I gate
        self.action_space = Discrete(len(GATES))

        # Observation space is a tuple indicating theta and phi region
        self.observation_space = Tuple(
            [Discrete(k), Discrete(k)]
        )

        # set seed for comparison
        self.seed(1)
        self.reset()

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        assert self.action_space.contains(action)
        self.last_action = action
        gate = action
        done = False
        
        # self.cir += GATES[gate] + f" q[0];\n"
        self.cir = GATES[gate] @ self.cir
        self.gate_counts[gate] += 1

        # have to check fidelity, other things anyway
        # qc = QuantumCircuit.from_qasm_str(self.cir)

        # discretize the bloch sphere here
        # sv = execute(qc, self.bend).result().get_statevector(qc)
        observation = statevector_to_bloch_reg(self.cir, self.k)

        if (observation == self.goal_reg):
            reward = 10 # - self.gate_counts[0] * 0.05 - self.gate_counts[1] * 0.05
            done = True
        else:
            reward = 0

        # if qc.depth() > self.depth_limit:
            
        #     reward = state_fidelity(goal_sv, const_sv)
        #     done = True
        #     theta = 2 * np.arccos(abs(const_sv[0]))
        #     phi = np.angle(const_sv[1])

        return (observation, reward, done, {})

    def reset(self):
        """Resets the environment to an initial state and returns an initial
        observation.
        Returns:
            observation (object): the initial observation.
        """
        self.last_action = None
        # self.cir = f"OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];"
        self.cir = np.array([1, 0])

        return ((0, 0), 0, False, {}) 

    def render(self, mode='human'):
        """Renders the environment.
        Args:
            mode: mode in which to display the circuit. Options include
                - human: render to the current display or terminal
                - rgb_array: Return an numpy.ndarray with shape (x, y, 3), 
                    representing RGB values for an image representation of the circuit
        Returns:
            rbg representation of the circuit or nothing
        """
        outfile = sys.stdout

        if (mode == 'human'):
            # print circuit
            c = QuantumCircuit.from_qasm_str(self.cir)
            print(c.draw())
        elif (mode == 'rgb_array'):
            # generate circuit image and return
            pass
        else:
            super(CircuitEnv, self).render(mode=mode)

    def seed(self, seed=None):
        """
        Sets the seed for this env's random number generator.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]