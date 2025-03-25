import numpy as np

import MDP
from FSCTrigger import *
from itertools import product
from mdpSolver import *
from GradientCal import *

def get_augMDP(mdp, trigger, sampledMDPs, adv_reward):
    """

    :param mdp: the original MDP.
    :param trigger: the finite-state trigger
    :param sampledMDPs:  the sampled set of Markov decision processes.
    :param adv_reward: player 1's reward function
    :return:
    """
    augmdp  = MDP.MDP()
    augstate = []
    K= len(sampledMDPs)
    augmdp.actlist = list(product(mdp.actlist, range(K)))
    augmdp.init = (mdp.init, trigger.trans[trigger.init][str(mdp.init)]) # assume deterministic initial states.
    augmdp.states = [augmdp.init]
    n= len(augmdp.states)
    augmdp.prob = {a: np.zeros((n,n)) for a in augmdp.actlist}
    pointer = 0
    prob_trans ={}
    reward = {}
    while pointer < len(augmdp.states):
        (s,m) = augmdp.states[pointer]
        for (a,k) in augmdp.actlist:
            reward[((s,m),(a,k))] = adv_reward[s][a] # adversarial reward is assigned
            for ns in mdp.states:
                if sampledMDPs[k].P(s,a,ns)>0:
                    p = sampledMDPs[k].P(s,a,ns)
                    trans = (s,a,ns)
                    nk = trigger.trans[m][trans]
                    if (ns, nk) not in augmdp.states:
                        augmdp.states.append((ns,nk))
                    ns_idx = augmdp.states.index((ns,nk))
                    prob_trans[(pointer,(a,k), ns_idx)]= p
        pointer += 1
    n = len(augmdp.states)
    augmdp.prob={a : np.zeros((n,n)) for a in augmdp.actlist}
    for trans in prob_trans.keys():
        augmdp.prob[trans[1]][trans[0],trans[2]]= prob_trans[trans]
    augmdp.reward = {state: {act: reward[(state,act)] for act in augmdp.actlist} for state in augmdp.states}
    return augmdp

def remove_ith_element(tup, i):
    return tup[:i] + tup[i+1:]

def marginalizedMDP(game, player, i):
    """
    :param game: multi-player Markov game
    :param player policy: policy for one player i
    :param i: the index of that player.
    :return: a game obtained after marginlization
    """
    acts = list(set([remove_ith_element(act, i)[0] for act in game.actlist]))
    acts_i = player.actlist
    mgame = MDP.MDP(game.init, acts, game.states.copy())
    n = len(game.states)
    mgame.prob = {a: np.zeros((n,n)) for a in mgame.actlist}
    for s in mgame.states:
        s_i = s[i]
        for act in game.actlist:
            for ns in mgame.states:
                act_wo_i = remove_ith_element(act, i)[0]
                temp = 0
                reward_temp = 0
                for ai in acts_i:
                    ai_idx = player.actlist.index(ai)
                    temp += game.P(s,act,ns)*player.policy[s_i][ai_idx]
                    reward_temp += game.reward[s][act]*player.policy[s_i][ai_idx] # R(s, a_j) = \sum_{a_i} R(s,a)* \pi(s,a_i)
                mgame.assign_P(s, act_wo_i, ns, temp)
                mgame.reward[s][act_wo_i] = reward_temp
    return mgame

def get_lowerbound(mdp, degrade_percent = 0.2):
    """

    :param mdp: a given mdp.
    :param degrade_percent: performance drop measured by percentage of the original optimal reward
    :return:
    """
    [V, pol] = valueIter(mdp)
    initial = mdp.states.index(mdp.init) # deterministic initial state.
    lb = V[initial]*(1-degrade_percent) # the lower bound on the performance.
    return lb


def get_joint_policy(augmdp, pol0, pol1):
    augmdp.act_len = len(augmdp.actlist)
    joint_policy = {s: np.zeros(augmdp.act_len) for s in augmdp.states}
    for joint_state in augmdp.states:
        for joint_action in augmdp.actlist:
            a0_idx = pol0.actlist.index(joint_action[0])
            a1_idx = pol1.actlist.index(joint_action[1])
            act_idx = augmdp.actlist.index(joint_action)
            joint_policy[joint_state][act_idx] = pol0.policy[joint_state[0]][a0_idx]*pol1.policy[joint_state[1]][a1_idx]
    jpolicy = Policy(augmdp.states.copy(), augmdp.actlist.copy(), False, joint_policy)
    return jpolicy

def covert_reward_pol_augmdp(augmdp, trigger_pol, adv_reward):
    aug_adv_reward  = {s: {a: 0 for a in augmdp.actlist} for s in augmdp.states}
    aug_policy = MDP.Policy(augmdp.states, trigger_pol.actlist, deterministic=False)
    for (s,m) in augmdp.states:
        for (a, k) in augmdp.actlist:
            aug_adv_reward[(s,m)][(a,k)] = adv_reward[s][a]
            aug_policy.policy[(s,m)][k] =  trigger_pol.policy[m][k]
    return aug_adv_reward, aug_policy


def covert_reward_pol_augmdp_pol0(augmdp, pol0, adv_reward):
    aug_adv_reward  = {s: {a: 0 for a in augmdp.actlist} for s in augmdp.states}
    aug_policy = MDP.Policy(augmdp.states, pol0.actlist, deterministic=False)
    for (s,m) in augmdp.states:
        for (a, k) in augmdp.actlist:
            a_idx = pol0.actlist.index(a)
            aug_adv_reward[(s,m)][(a,k)] = adv_reward[s][a]
            aug_policy.policy[(s,m)][a_idx] =  pol0.policy[s][a_idx]
    return aug_adv_reward, aug_policy


def softmax(x, tau):
    exp_x = np.exp((x - np.max(x))) / tau  # Subtract max(x) for numerical stability
    return exp_x / np.sum(exp_x)

def theta_to_policy(mdp, theta, tau):
    theta_dict = {s:np.zeros(mdp.act_len) for s in mdp.states}
    st_len = len(mdp.states)
    for i in range(st_len):
        s = mdp.states[i]
        theta_dict[s] = theta[i * mdp.act_len : i*mdp.act_len+mdp.act_len]
    policy = {s : softmax(theta_dict[s], tau) for s in mdp.states}
    return policy

def policy_to_theta(mdp, pol):
    """
    :param mdp:
    :param pol: policy class
    :return:
    """
    V = policyEval(mdp, pol)
    # calculate the Q value
    i=0
    mdp.act_len= len(mdp.actlist)
    theta_dict = {s: np.zeros(mdp.act_len) for s in mdp.states}
    theta = np.zeros(len(mdp.states)*mdp.act_len)
    for s in mdp.states:
        s_idx = mdp.states.index(s)
        theta_dict[s] = [mdp.reward[s][a]+ mdp.gamma*mdp.prob[a][s_idx,:].dot(V) for a in mdp.actlist]
        theta[i: i+mdp.act_len] = theta_dict[s]
        i = i+mdp.act_len
    return theta

import matplotlib.pyplot as plt
import GradientCalTrigger

def switchingGradient(mdp, adv_reward, trigger, augmdp, K,  episodes=1000, learning_rate=0.01, max_iters=200, tolerance=1e-2):
    original_states = mdp.states
    original_actlist = mdp.actlist
    trigger_states = trigger.memory_space
    trigger_acts = list(range(K))
    pol0 = Policy(original_states, original_actlist, False) # randomized policy, initialized to a uniform random one.
    pol1 = Policy(trigger_states, trigger_acts, False)
    theta0 = policy_to_theta(mdp, pol0)
    lb= get_lowerbound(mdp, degrade_percent =0.1)
    initial_joint = augmdp.states.index(augmdp.init)
    initial_0 = mdp.states.index(mdp.init)
    epsilon = 1e-6  # Convergence threshold
    tau = 0.1 # temperature
    learning_rate2= 0.0001
    GradientCal0 = GradientCal(mdp, theta0, tau , mdp.reward)
    marg_MDP1 = marginalizedMDP(augmdp, pol0, 0)
    aug_adv_reward, aug_pol1= covert_reward_pol_augmdp(augmdp, pol1, adv_reward)
    theta1 = np.zeros(len(trigger.memory_space)*K)
    V0_iter= []
    V1_iter= []
    sample_size = 50
    for episode in range(episodes):
        V0 = policyEval(mdp, pol0)
        if V0[mdp.states.index(mdp.init)] < lb: # performance is worse than lower bound, constraint is violated.
            # gradient ascent for reward 0
            #print("The constraint is violated: V0 is", V0[mdp.init])
            samples = mdp.generate_samples(pol0, sample_size)
            GradientCal0 = GradientCal(mdp, pol0, tau, mdp.reward)
            grad_pol0 = GradientCal0.dJ_dtheta(samples) # gradient ascent one step in the original MDP.
            if np.linalg.norm(grad_pol0, ord=np.inf) < tolerance:  # Stop if gradient is too small
                break
            theta0 += learning_rate * grad_pol0  # Update step
            pol0 = Policy(mdp.states, mdp.actlist, False, theta_to_policy(mdp, theta0, tau))
        else:
            #print("The constraint is satisfied: V0 is", V0[mdp.init])
            marg_MDP0 = marginalizedMDP(augmdp, pol1, 1) # marginalize out the policy for trigger
            aug_adv_reward0, aug_policy0 = covert_reward_pol_augmdp_pol0(augmdp, pol0, adv_reward) # compute the policy for player 1 in the augmented state space
            samples0 = marg_MDP0.generate_samples(aug_policy0, sample_size) # obtain samples from the augmented MDP.
            GradientCal1_0 = GradientCalTrigger.GradientCalTrigger(marg_MDP0, 0, pol0,  tau, adv_reward)
            grad_pol0_1 = GradientCal1_0.dJ_dtheta(samples0)  # gradient ascent one step in the marginalized MDP with policy 1.
            theta0 += learning_rate2 * grad_pol0_1  # Update step
            pol0 = Policy(mdp.states, mdp.actlist, False, theta_to_policy(mdp, theta0, tau))
            # compute the gradient for policy 1
            marg_MDP1 = marginalizedMDP(augmdp, pol0, 0)  # marginalize out the policy for the backdoor policy
            aug_adv_reward1, aug_policy1 = covert_reward_pol_augmdp(augmdp,  pol1, adv_reward)
            samples1 = marg_MDP1.generate_samples(aug_policy1, sample_size) # obtain samples from the augmented MDP.
            GradientCal1_1 = GradientCalTrigger.GradientCalTrigger(marg_MDP1, 1, pol1, tau, adv_reward)
            grad_pol1_1 = GradientCal1_1.dJ_dtheta(samples1)  # gradient ascent one step in the marginalized MDP with policy 1.
            theta1 += learning_rate2 * grad_pol1_1  # Update step
            if np.linalg.norm(grad_pol0_1, ord=np.inf) < tolerance and np.linalg.norm(grad_pol1_1, ord=np.inf) < tolerance:  # Stop if gradient is too small
                break
            V1 = policyEval(augmdp, get_joint_policy(augmdp, pol0, pol1))
            V1_iter.append(V1[augmdp.states.index(augmdp.init)])
            print("The value of the attacker's MDP under trigger:", V1[augmdp.states.index(augmdp.init)])
            #input("Press Enter to continue...")
            V0 =  policyEval(mdp, pol0)
            V0_iter.append(V0[mdp.states.index(mdp.init)])
            print("The value for the original MDP:", V0[mdp.states.index(mdp.init)])
            #   if episode % 100 == 0:
            #       print(f"Episode {episode}: ")
    plt.plot(V0_iter)
    plt.plot(V1_iter)
    plt.show()
    return
