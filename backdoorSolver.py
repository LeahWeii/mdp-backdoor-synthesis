import random

import Sample
from MDPgenerator import *
import math
import numpy as np
import sys

def get_augMDP(mdp, trigger, sampledTrans):
    """

    :param mdp: the original MDP.
    :param trigger: the finite-state trigger
    :param sampledTrans:  the sampled set of transition functions.
    :return:
    """

def update_trans_attacker(mdp, mdp_attacker, trans_samples, pi0):
    #pi0[s] = [p1,p2,p3,...]
    states = mdp_attacker.states
    actlist = mdp.actlist

    for s in states:
        mdp_attacker.trans[s] = {}
        for a1 in mdp_attacker.actlist:
            mdp_attacker.trans[s][a1] = {}
            for ns in states:
                mdp_attacker.trans[s][a1][ns] = 0
                for a0 in actlist:
                    mdp_attacker.trans[s][a1][ns] += pi0[s][actlist.index(a0)] * trans_samples[mdp_attacker.actlist.index(a1)][s][a0][ns]

def update_trans(mdp, trans_list, pi1):
    #pi1[s] = [p1, p2, ...], assume we use markov trigger
    trans = dict()
    for s in mdp.states:
        trans[s] = {}
        for a in mdp.actlist:
            trans[s][a] = {}
            for ns in mdp.states:
                probs = [
                    d[s][a][ns]
                    for d in trans_list
                ]
                trans[s][a][ns] = sum(a * b for a, b in zip(pi1[s], probs))
    mdp.trans = trans

def print_trans(mdp):
    print("\nTrans:")
    for action in mdp.actlist:
        print(f"Action '{action}':")
        for state_from in mdp.states:
            for state_to in mdp.states:
                trans = mdp.trans[state_from][action][state_to]
                print(f"  From state '{state_from}' to state '{state_to}' with probability {trans:.2f}")
    for s in mdp.trans:
        for a in mdp.trans[s]:
            prob_sum = sum(mdp.trans[s][a].values())
            if not math.isclose(prob_sum, 1.0, rel_tol=1e-9):
                print(f"Sample {i}, State {s}, Action {a}: Sum = {prob_sum}")
    print("All transition probabilities sum to 1 for each state-action pair.")

def theta_to_policy(mdp,theta):
    states = mdp.states
    actions = mdp.actlist
    tau = 1
    index = 0
    theta_dict = {}
    for state in states:
        theta_dict[state] = {}
        for action in actions:
            theta_dict[state][action] = theta[index]
            index += 1

    policy = {}
    for state in states:
        # Get maximum theta value for numerical stability
        max_theta = max(theta_dict[state].values())

        # Calculate exponential values for all actions in order
        exp_vals = [
            math.exp((theta_dict[state][action] - max_theta) / tau)
            for action in actions
        ]

        # Calculate normalization sum
        exp_sum = sum(exp_vals)

        # Create probability distribution ordered by actions
        policy[state] = [val / exp_sum for val in exp_vals]

    for st in states:
        sum_of_prob = sum(policy[st])
        if sum_of_prob > 1.000001 or sum_of_prob < 0.999999:
            sys.exit("policy is not well defined")
    return policy


def hardmax_to_softmax(hardmax_policy, epsilon=1e-6):
    """
    Convert a hardmax policy (one-hot per state) into a softmax policy.

    Args:
        hardmax_policy: Dict[state, Dict[action, float]]  (one-hot probabilities)
        epsilon: Small smoothing factor to avoid division by zero

    Returns:
        Dict[state, Dict[action, float]]: Softmax probabilities
    """
    softmax_policy = {}
    for state, action_probs in hardmax_policy.items():
        # Convert to probabilities with smoothing and renormalize
        smoothed_probs = np.array([p + epsilon for p in action_probs.values()])
        normalized_probs = smoothed_probs / smoothed_probs.sum()

        # Rebuild the action probability dictionary
        softmax_policy[state] = {
            action: normalized_probs[i]
            for i, action in enumerate(action_probs.keys())
        }

    return softmax_policy

def softmax_policy_to_theta(policy):
    #policy is dict[s][a]
    # Ensure policy is a numpy array
    policy = np.array([p for state_probs in policy.values() for p in state_probs.values()])
    # Calculate theta by taking the log of policy
    # We subtract the max value to avoid numerical instability
    theta = np.log(policy) - np.max(np.log(policy))
    return theta

def GD(trans_list, mdp0,mdp,mdp_attacker):
    init = mdp0.init
    n=10
    num_pairs = 100
    v_opt, policy0_feasible = mdp0.value_iteration()
    v_opt = np.dot(v_opt,init)
    epsilon = 0.1 * v_opt

    policy0_feasible_softmax = hardmax_to_softmax(policy0_feasible)
    theta0 = softmax_policy_to_theta(policy0_feasible_softmax) #start from a feasible theta0
    theta1 = np.zeros(mdp_attacker.theta_size)

    ref_v = math.inf
    count = 0
    objs = []
    while True:
        policy0 = theta_to_policy(mdp0,theta0)
        v = mdp0.policy_evaluation(policy0)
        v = np.dot(v,init)
        if v>v_opt - epsilon: #feasible
            policy1 = theta_to_policy(mdp_attacker, theta1)
            update_trans(mdp, trans_list, policy1)
            sample_mdp = Sample.SampleTraj(mdp)
            sample_mdp.generate_traj(n, num_pairs, policy0)  # for V1
            g = mdp.dJ_dtheta(sample_mdp, policy0)
            theta0 += 1 * math.exp(-0.0001 * (count - 1))*g

            update_trans_attacker(mdp, mdp_attacker, trans_list, policy0)
            g_attacker = mdp_attacker.dJ_dtheta(sample_mdp, policy1)  # if num_trans<num_actions, there will be error due to reward is only state dependent
            theta1 += 0.0001* math.exp(-0.0001 * (count - 1))*g_attacker

        else: #violation
            policy0 = theta_to_policy(mdp0, theta0)
            sample_mdp0 = Sample.SampleTraj(mdp0)
            sample_mdp0.generate_traj(n, num_pairs, policy0)
            g0 = mdp0.dJ_dtheta(sample_mdp0, policy0)
            theta0 += 1* math.exp(-0.0001 * (count - 1)) * g0

        update_trans(mdp, trans_list, policy1)
        new_v = mdp.policy_evaluation(policy0)
        new_v = np.dot(new_v, init)
        if count%10 ==0:
            print(count)
        count += 1
        if abs(new_v-ref_v)<1e-7 or count>=10000:
            break
        else:
            ref_v = new_v
            objs.append(ref_v)

    np.save('results/objs.npy',np.array(objs))



if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    trans_list, mdp0, mdp, mdp_attacker = get_aug_mdp(3,3,3, 0.5)
    with open('saved_objects.pkl', 'wb') as f:
        pickle.dump(trans_list, f)
        pickle.dump(mdp0, f)
        pickle.dump(mdp, f)
        pickle.dump(mdp_attacker, f)

    GD(trans_list, mdp0,mdp,mdp_attacker)

    # pi = {}
    # for state in mdp.states:
    #     # Generate random probabilities for each action
    #     probs = [random.random() for _ in mdp.actlist]
    #
    #     # Normalize probabilities to sum to 1
    #     total = sum(probs)
    #     normalized_probs = [p / total for p in probs]
    #
    #     # Assign normalized probabilities to the state
    #     pi[state] = normalized_probs
    #
    # update_trans_attacker(mdp, mdp_attacker, trans, pi)
    #
    # update_trans(mdp,trans, pi)
    #
    # print_trans(mdp_attacker)




