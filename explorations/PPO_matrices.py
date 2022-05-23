import gym
import gym.spaces
import numpy as np
from scipy.interpolate import interp1d
from stable_baselines.common.policies import ActorCriticPolicy, MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc
import tensorflow as tf

class OneQEnv(gym.Env):
    def __init__(self, gamma=0.8, max_steps=50, qubits=3):
        self.interp = interp1d([-1.001, 1.001], [0, 255])
        self.qubits = qubits
        # self.goal = bell_state(self.qubits)
        self.goal = goal_state()
        # discount factor
        self.gamma = gamma
        # identify the observation and action spaces
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(2**self.qubits, 2**self.qubits, 2), dtype=float)
        self._actions = gates
        self.action_space = gym.spaces.Discrete(len(self._actions))
        # the state will be the wavefunction probs
        p = Program(self.qubits)
        for i in range(self.qubits):
            p.inst(('I', i))
        self._program = p
        self._wfn = self._program.simulate()

        dm = np.outer(self._wfn, self._wfn)
        self.state = self.interp(np.moveaxis(np.stack([dm.real, dm.imag], axis=0), 0, 2))
        self.current_step = 0
        self.max_steps = max_steps
        self.info = {}
        
    def step(self, action):
        gate = self._actions[action]
        self._program.inst(gate)
        self._wfn = self._program.simulate()
        dm = np.outer(self._wfn, self._wfn)
        self.state = self.interp(np.moveaxis(np.stack([dm.real, dm.imag], axis=0), 0, 2))
        self.current_step += 1

        reward = abs(self._wfn.T.conj() @ self.goal)**2
        if reward > 0.8:
            done = True
            # reward = 1
        elif self.current_step >= self.max_steps:
            done = True
            # reward = 0
        else:
            done = False
            # reward = 0

        return self.state, reward, done, self.info
    
    def reset(self):
        p = Program(self.qubits)
        for i in range(self.qubits):
            p.inst(('I', i))
        self._program = p
        self._wfn = self._program.simulate()
        dm = np.outer(self._wfn, self._wfn)
        self.state = self.interp(np.moveaxis(np.stack([dm.real, dm.imag], axis=0), 0, 2))
        self.current_step = 0
        
        return self.state

def custom_cnn(scaled_images, **kwargs):
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=8, filter_size=2, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=16, filter_size=2, stride=2, init_scale=np.sqrt(2), **kwargs))
    # layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_2)
    layer_4 = activ(linear(layer_3, 'fc1', n_hidden=128, init_scale=np.sqrt(2)))
    return activ(linear(layer_4, 'fc2', n_hidden=64, init_scale=np.sqrt(2)))

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None, net_arch=None,
                 act_fun=tf.tanh, cnn_extractor=custom_cnn, feature_extraction="cnn", **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)

        with tf.variable_scope("model", reuse=reuse):
            if feature_extraction == "cnn":
                pi_latent = vf_latent = cnn_extractor(self.processed_obs, **kwargs)

            self._value_fn = linear(vf_latent, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


class Program():
    def __init__(self, qubits):
        self._qubits = qubits
        self._instructions = []
        self._inststr = []
        self._init = np.eye(2**self._qubits)[0]

    def inst(self, *instructions) -> "Program":
        for instruction in instructions:
            if isinstance(instruction, list):
                self.inst(*instruction)
            elif isinstance(instruction, tuple):
                if len(instruction) < 2:
                    raise ValueError("tuple should have at least two elements")
                elif len(instruction) == 2:
                    self._instructions.append(GATES[instruction[0]](instruction[1]))
                else:
                    self._instructions.append(GATES[instruction[0]](instruction[1], instruction[2]))
                self._inststr.append(instruction)

        return self

    def simulate(self) -> np.array:
        _ = self._init
        for g in self._instructions:
            _ = g @ _
        return _

    def __len__(self) -> int:
        return len(self._instructions)

    def __str__(self) -> str:
        """
        A string representation of the matrix program
        """
        return "\n".join([' '.join([str(i) for i in tup]) for tup in self._inststr])

    # def dagger(self):
    #     pass
    # 
    # def pop(self) -> np.array:
    #     res = self._instructions.pop()
    #     return res

qubits = 3
SPECIAL_CONTROL_ONE = (1+0j)

GATES = {
    'I': lambda qb: tensor(np.array([[1+0j, 0j], [0j, 1+0j]]), qb),
    'H': lambda qb: tensor(1/np.sqrt(2) * np.array([[1+0j, 1+0j], [1+0j, -1+0j]]), qb),
    'X': lambda qb: tensor(np.array([[0j, 1+0j], [1+0j, 0j]]), qb),
    'Y': lambda qb: tensor(np.array([[0j, -1j], [1j, 0j]]), qb),
    'Z': lambda qb: tensor(np.array([[1+0j, 0j], [0j, -1+0j]]), qb),
    'RX': lambda theta, qb: tensor(np.array([[np.cos(theta/2)+0j, -1j*np.sin(theta/2)], [-1j*np.sin(theta/2), np.cos(theta/2)+0j]]), qb),
    'RY': lambda theta, qb: tensor(np.array([[np.cos(theta/2)+0j, -np.sin(theta/2)+0j], [np.sin(theta/2)+0j, np.cos(theta/2)+0j]]), qb),
    'RZ': lambda phi, qb: tensor(np.array([[np.cos(phi/2) - 1j*np.sin(phi/2), 0j], [0j, np.cos(phi/2) + 1j*np.sin(phi/2)]]), qb),
    'CNOT': lambda ctrl, tgt: controlled_tensor(np.array([[0, 1], [1, 0]]), ctrl, tgt) # issue with imaginary numbers
}

def tensor(gate, qubit):
    seq = [np.eye(2) for i in range(qubits)]
    
    if qubit >= qubits:
        raise ValueError("Qubit out of range")

    seq[qubit] = gate
    if len(seq) == 1:
        return gate
    elif len(seq) == 2:
        return np.kron(seq[1], seq[0])
    else:
        kron_matrix = np.kron(seq[1], seq[0])
        for s in seq[2:]:
            kron_matrix = np.kron(s, kron_matrix)
        return kron_matrix
        

def controlled_tensor(gate, control, target):
    seq = [np.eye(2) for i in range(qubits)]
    
    if len(seq) == 1:
        raise ValueError("Must have at least two qubits to perform a controlled gate")
    if target >= qubits or control >= qubits:
        raise ValueError("Target or control qubit out of range")

    seq[target] = gate
    seq[control] = [[SPECIAL_CONTROL_ONE, 0],[0,1]]
    
    if len(seq) == 2:
        return np.array(kronecker_product(seq[1], seq[0])).real
    else:
        kron_matrix = kronecker_product(seq[1], seq[0])
        for s in seq[2:]:
            kron_matrix = kronecker_product(s, kron_matrix)
        return np.array(kron_matrix).real # clipping the imaginary part -- could be a problem if doing C-RY, CY, etc.

def kronecker_product(m1, m2):
    w1, h1 = len(m1), len(m1[0])
    w2, h2 = len(m2), len(m2[0])
    return [[
        controlled_product(m1[i1][j1], m2[i2][j2], i1, i2, j1, j2)
        for i1 in range(w1) for i2 in range(w2)]
        for j1 in range(h1) for j2 in range(h2)]

def controlled_product(v1, v2, i1, i2, j1, j2):
    if v1 is SPECIAL_CONTROL_ONE:
        return SPECIAL_CONTROL_ONE if i2==j2 else 0
    if v2 is SPECIAL_CONTROL_ONE:
        return SPECIAL_CONTROL_ONE if i1==j1 else 0
    return v1*v2

def bell_state(qb):
    p = Program(qb)
    p.inst(('RY', np.pi/2, 0))
    for i in range(qb-1):
        p.inst(('CNOT', i, i+1))
    wfn = p.simulate()
    dm = np.outer(wfn, wfn)
    state = np.moveaxis(np.stack([dm.real, dm.imag], axis=0), 0, 2)
    interp = interp1d([-1.001, 1.001], [0, 255])
    return interp(state)

def goal_state():
    wfn = np.array([ 0.00000000e+00+0.j        ,  3.46635318e-01+0.31266666j,
        3.95516953e-16-0.62533332j,  0.00000000e+00+0.j        ,
       -7.00828284e-16-0.62533332j,  0.00000000e+00+0.j        ,
        0.00000000e+00+0.j        ,  0.00000000e+00+0.j        ])
    dm = np.outer(wfn, wfn)
    state = np.moveaxis(np.stack([dm.real, dm.imag], axis=0), 0, 2)
    interp = interp1d([-1.001, 1.001], [0, 255])
    return wfn

if __name__ == "__main__":
    num_angles = 32
    angles = np.linspace(0.0, 2*np.pi, num_angles)
    gates = [('CNOT', 0, 1), ('CNOT', 1, 0), ('CNOT', 1, 2), ('CNOT', 2, 1), ('CNOT', 0, 2), ('CNOT', 2, 0)]
    gates += [('RX', theta, q) for theta in angles for q in range(qubits)]
    gates += [('RZ', theta, q) for theta in angles for q in range(qubits)]
    gates += [('H', q) for q in range(qubits)]

    env = OneQEnv()
    env_vec = DummyVecEnv([lambda: env])

    model = PPO2(CustomPolicy, env_vec, verbose=1)
    model.learn(total_timesteps=500000)

    def wfn_to_dm(wfn):
        dm = np.outer(wfn, wfn)
        state = np.moveaxis(np.stack([dm.real, dm.imag], axis=0), 0, 2)
        interp = interp1d([-1.001, 1.001], [0, 255])
        return interp(state)

    goal_wfn = np.array([ 0.00000000e+00+0.j        ,  3.46635318e-01+0.31266666j,
        3.95516953e-16-0.62533332j,  0.00000000e+00+0.j        ,
       -7.00828284e-16-0.62533332j,  0.00000000e+00+0.j        ,
        0.00000000e+00+0.j        ,  0.00000000e+00+0.j        ])

    done = False
    env.reset()
    prog = Program(3)
    prog.inst(('I', 0)).inst(('I', 1)).inst(('I', 2))
    wfn = prog.simulate()
    obs = wfn_to_dm(wfn)

    while not done:
        optimal_action, _ = model.predict(obs)
        print(' '.join([str(i) for i in gates[optimal_action]]))
        prog.inst(gates[optimal_action])
        obs, rewards, done, info = env.step(optimal_action)
        
    wfn = prog.simulate()
    print(f"Wavefunction: {wfn}")
    print(f"Fidelity: {np.abs(np.conj(wfn) @ goal_wfn)**2}")
