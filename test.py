import numpy as np
import random
from collections import defaultdict
import json
from HT_circuit_env import CircuitEnv

# env = CircuitEnv(n=10**4)
# for i_episode in range(1):
#     observation = env.reset()
#     for t in range(10):
#         env.render()
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         print(observation)
#         if done:
#             print(reward)
#             # print("Episode finished after {} timesteps".format(t+1))
#             break
#         # time.sleep(0.5)
# env.close()

def random_building():
    q = defaultdict(lambda: [0,0,0])
    # with open('qtable.json') as f:
    #     prev_q = json.load(f)
    # for key in prev_q:
    #     q[key] = prev_q[key]
    rewards = np.array([])

    # Hyperparameters
    epsilon = 1
    epsilon_min = 0.05
    epsilon_decay = 0.999993
    gamma = 0.9
    alpha = 0.1

    def updateQ(s, a, new_s, r):
        q_value = q[str(s)]
        q_value[a] = ((1 - alpha) * q_value[a]) + (alpha * (r + (gamma * max(q[str(new_s)]))))
        q[str(s)] = q_value

    n = 10**8
    env = CircuitEnv(n=n, k=32)
    print(env.goal_reg)

    for i_episode in range(10000):
        observation, _, _, _ = env.reset()

        for t in range(2 * 10**1): # number of gates to place in one episode
            if (random.uniform(0, 1) > epsilon):
                action = np.argmax(q[str(observation)])
            else:
                action = env.action_space.sample()

            old_s = observation
            observation, reward, done, info = env.step(action)
            # rewards = np.append(max(0, reward - t/n), reward)

            # updateQ(old_s, action, observation, max(0, reward - t/n))
            updateQ(old_s, action, observation, max(0, reward))

            
            if done:
                break

        if (epsilon > epsilon_min):
            epsilon *= epsilon_decay

    fo = open('qtable.json', 'w')
    json.dump(q, fo)
    fo.close()
    env.close()

random_building()