import random
import numpy as np

class DeepQNetwork():
    """Q-Learning with deep neural network to learn the control policy. 
    Uses a deep neural network model to predict the expected utility (Q-value) of executing an action in a given state. 
    Reference: https://arxiv.org/abs/1312.5602
    Parameters:
    -----------
    env: Env 
        class that represents the quantum state
    epsilon: float
        The epsilon-greedy value. The probability that the agent should select a random action instead of
        the action that will maximize the expected utility. 
    gamma: float
        Determines how much the agent should consider future rewards. 
    decay_rate: float
        The rate of decay for the epsilon value after each epoch.
    min_epsilon: float
        The value which epsilon will approach as the training progresses.
    """
    def __init__(self, env, epsilon=1, gamma=0.9, decay_rate=0.005, min_epsilon=0.1):
        self.epsilon = epsilon
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        self.memory_size = 300
        self.memory = []

        # Initialize the environment
        self.env = gym.make(env_name)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

    