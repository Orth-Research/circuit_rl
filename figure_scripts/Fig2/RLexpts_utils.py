import numpy as np
import sys


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


def policy_iteration(env, d_raw_counts, gamma=0.8, theta=1e-8):
    policy = np.ones((env.nS, env.nA)) / env.nA
    counter_iter = 0
    while True:
        counter_iter += 1
        print(f"\rCounter for policy iteration: {counter_iter}", end="")
        sys.stdout.flush()
        V = policy_evaluation(env, policy, d_raw_counts, gamma=gamma, theta=theta)
        improved_policy = policy_improvement(env, V, d_raw_counts, gamma=gamma)
        if np.max(np.abs(policy - improved_policy)) < theta:
            break
        policy = improved_policy
    return policy


def policy_iteration_v2(env, d_raw_counts, gamma=0.8, theta=1e-8):
    policy = np.ones((env.nS, env.nA)) / env.nA
    counter_iter = 0
    while True:
        counter_iter += 1
        print(f"\rCounter for policy iteration: {counter_iter}", end="")
        sys.stdout.flush()
        V = policy_evaluation(env, policy, d_raw_counts, gamma=gamma, theta=theta)
        improved_policy = policy_improvement(env, V, d_raw_counts, gamma=gamma)
        if np.max(np.abs(policy - improved_policy)) < theta:
            break
        policy = improved_policy
    return policy, counter_iter
