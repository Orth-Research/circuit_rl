import numpy as np
from gym.utils import seeding
from gym.spaces import Discrete, Tuple, Box
import gym
from qiskit.quantum_info import state_fidelity
from qiskit import *

CIRCUITS = {}
GATES = {
    0: 'h',
    1: 't'
    # 0: 'u3(1.57079632679490,-1.57079632679490,1.57079632679490)',
    # 1: 'u3(1.57079632679490,0,0)',
    # 2: 'u3(1.57079632679490,-0.785398163397448,0.785398163397449)',
    # 3: 'cx'
}

def generate_random_circuit(m=1, n=3):
    """Generates a random supremacy-style circuit using a specific set of gates.
    Code modified from https://arxiv.org/pdf/2005.10811.pdf
    Args:
        m: number of cycles to generate
        n: number of qubits
    Returns:
        s: qasm string of the generated circuit
    """
    
    sqrtx = 'u3(1.57079632679490,-1.57079632679490,1.57079632679490)'
    sqrty = 'u3(1.57079632679490,0,0)'
    sqrtw = 'u3(1.57079632679490,-0.785398163397448,0.785398163397449)'
    gates = [sqrtx, sqrty, sqrtw]
    
    s = f"OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[{n}];"

    def rand_cx(n):
        q1 = np.random.randint(n)
        q2 = np.delete(np.arange(n), q1)
        q2 = np.random.choice(q2, size=1)[0]
        return f"cx q[{q1}], q[{q2}];\n"
    
    last_gates = -np.ones(n, dtype=np.int64)
    for i in range(m):
        # single-qubit gates
        for j in range(n):
            choices = np.arange(len(gates))
            if last_gates[j] != -1:
                choices = np.delete(choices, last_gates[j])
            g = np.random.choice(choices, size=1)[0]
            last_gates[j] = g
            s += gates[g] + ' q[' + str(j) + '];\n'
        
        # two-qubit gate
        # g, q1, q2 = rand_cx(n)
        # s += g
        # last_gates[q1] = -1
        # last_gates[q2]
    return s

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
    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.
    The methods are accessed publicly as "step", "reset", etc...
    """

    metadata = {'render.modes': ['human', 'rgb_array']}
    reward_range = (-float('inf'), float('inf'))

    def __init__(self, cir_name=None, depth_limit=4, qb=2):
        if cir_name is None:
            qc = generate_random_circuit(m=depth_limit, n=qb)
            self.goal_cir = QuantumCircuit.from_qasm_str(qc)
            print(self.goal_cir)
        else:
            self.goal_cir = QuantumCircuit.from_qasm_str(CIRCUITS[cir_name])

        self.qb = qb
        self.depth_limit = depth_limit
        self.bend = Aer.get_backend('statevector_simulator')
        self.cir = f"OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[{qb}];"

        # Two subactions
        #   1. Which qubit to add gate to
        #   2. Which gate to append (CNOT, sqrtx/y/w)
        #   3. Control qubit (ignored if gate isn't CNOT)
        self.action_space = Tuple(
            [Discrete(self.qb), Discrete(len(GATES)), Discrete(self.qb)]
        )
        # Observation space is a RGB representation of the circuit, subject to change
        # self.observation_space = Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
        self.observation_space = Discrete(1)

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
        qb, gate, target = action
        done = False
        
        if (GATES[gate] == 'cx'):
            if (qb != target):
                self.cir += GATES[gate] + f" q[{qb}], q[{target}];\n"
        else:
            self.cir += GATES[gate] + f" q[{qb}];\n"

        # have to check fidelity, other things anyway
        qc = QuantumCircuit.from_qasm_str(self.cir)
        reward = 0

        if qc.depth() > self.depth_limit:
            goal_sv = execute(self.goal_cir, self.bend).result().get_statevector(self.goal_cir)
            const_sv = execute(qc, self.bend).result().get_statevector(qc)
            reward = state_fidelity(goal_sv, const_sv)
            done = True

        return (0, reward, done, {})

    def reset(self):
        """Resets the environment to an initial state and returns an initial
        observation.
        Returns:
            observation (object): the initial observation.
        """
        self.last_action = None
        self.cir = f"OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[{self.qb}];"

        return (0, 0, False, {}) 

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