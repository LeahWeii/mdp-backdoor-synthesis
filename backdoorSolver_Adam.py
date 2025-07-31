import numpy as np

import MDP
from FSCTrigger import *
from itertools import product
from mdpSolver import *
from GradientCal import *
import os


def get_augMDP(mdp, trigger, sampledMDPs, adv_reward, adv_cost = None):
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
    augmdp.init = (mdp.init, trigger.trans[trigger.init][str([mdp.init])]) # assume deterministic initial states.
    augmdp.states = [augmdp.init]
    n= len(augmdp.states)
    augmdp.prob = {a: np.zeros((n,n)) for a in augmdp.actlist}
    pointer = 0
    prob_trans ={}
    reward = {}
    agent_reward = {}
    alpha = 0
    while pointer < len(augmdp.states):
        (s,m) = augmdp.states[pointer]
        for (a,k) in augmdp.actlist:
            reward[((s,m),(a,k))] = adv_reward[s][a] - alpha*adv_cost[m][k] # adversarial reward is assigned
            agent_reward[((s,m),(a,k))] = mdp.reward[s][a] # agent's reward is assigned
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
    augmdp.agent_reward = {state: {act: agent_reward[(state,act)] for act in augmdp.actlist} for state in augmdp.states}
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
    act_len = len(mdp.actlist)
    theta_dict = {s:np.zeros(act_len) for s in mdp.states}
    st_len = len(mdp.states)
    for i in range(st_len):
        s = mdp.states[i]
        theta_dict[s] = theta[i * act_len : i* act_len+ act_len]
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

def plot_results(path, V0_iter, V1_iter, V0_iter_attack, episodes, lb):
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    # First subplot: V0_iter with lower bound
    axs[0].plot(range(episodes), V0_iter, label='V0')
    axs[0].axhline(y=lb, color='r', linestyle='--', linewidth=2, label=f'Lower bound: {lb:.2f}')
    axs[0].set_xlabel('Episodes')
    axs[0].set_ylabel('Value')
    axs[0].set_title('V0 Iteration Results')
    axs[0].legend()
    axs[0].grid(True)
    # Second subplot: V1_iter
    axs[1].plot(range(episodes), V1_iter, label='V1', color='orange')
    axs[1].set_xlabel('Episodes')
    axs[1].set_ylabel('Value')
    axs[1].set_title('V1 Iteration Results')
    axs[1].legend()
    axs[1].grid(True)
    # Third subplot: V0_iter_attack
    axs[2].plot(range(episodes), V0_iter_attack, label='V0 under attack', color='green')
    axs[2].set_xlabel('Episodes')
    axs[2].set_ylabel('Value')
    axs[2].set_title('V0 under Attack Iteration Results')
    axs[2].legend()
    axs[2].grid(True)
    plt.tight_layout()
    plt.savefig(f"{path}/plot_v_iter.png")
    # plt.show()

import GradientCalTrigger



def plot_from_pickled_results(path, episodes=None, lb=None):
    """
    Read V0_iter, V1_iter, V0_attack from pickled files in path and plot them in three subplots.
    If episodes or lb are not provided, they are inferred from the data or skipped.
    """
    # Load pickled results
    with open(os.path.join(path, "V0.pkl"), "rb") as f0:
        V0_iter = pickle.load(f0)
    with open(os.path.join(path, "V1.pkl"), "rb") as f1:
        V1_iter = pickle.load(f1)
    with open(os.path.join(path, "V0_attack.pkl"), "rb") as f2:
        V0_iter_attack = pickle.load(f2)
    if episodes is None:
        episodes = min(len(V0_iter), len(V1_iter), len(V0_iter_attack))
    # Plot
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    axs[0].plot(range(episodes), V0_iter[:episodes], label='V0')
    if lb is not None:
        axs[0].axhline(y=lb, color='r', linestyle='--', linewidth=2, label=f'Lower bound: {lb:.2f}')
    axs[0].set_xlabel('Episodes')
    axs[0].set_ylabel('Value')
    axs[0].set_title('V0 Iteration Results')
    axs[0].legend()
    axs[0].grid(True)
    axs[1].plot(range(episodes), V1_iter[:episodes], label='V1', color='orange')
    axs[1].set_xlabel('Episodes')
    axs[1].set_ylabel('Value')
    axs[1].set_title('V1 Iteration Results')
    axs[1].legend()
    axs[1].grid(True)
    axs[2].plot(range(episodes), V0_iter_attack[:episodes], label='V0 under attack', color='green')
    axs[2].set_xlabel('Episodes')
    axs[2].set_ylabel('Value')
    axs[2].set_title('V0 under Attack Iteration Results')
    axs[2].legend()
    axs[2].grid(True)
    plt.tight_layout()
    plt.show()

def switchingGradient(mdp,epsilon, adv_reward, trigger, augmdp, K, path,  episodes=5000, lr=0.1, max_iters=200, tolerance=1e-2):
    original_states = mdp.states
    original_actlist = mdp.actlist
    trigger_states = trigger.states
    trigger_acts = list(range(K))
    pol0 = Policy(original_states, original_actlist, False) # randomized policy, initialized to a uniform random one.
    pol1 = Policy(trigger_states, trigger_acts, False)
    theta0 = policy_to_theta(mdp, pol0)
    lb= get_lowerbound(mdp, degrade_percent =epsilon)
    tau = 0.1 # temperature
    theta1 = np.zeros(len(trigger.states)*K)
    V0_iter= []
    V0_iter_attack = []
    V1_iter= []
    sample_size = 5
    V_original, pol0_opt =   valueIter(mdp, tau)
    V_attack  = policyEval(mdp,  pol0_opt, 0.01, adv_reward)

    V_0 = policyEval(mdp,  pol0_opt, 0.01)
    print("The value of opgimal policy for attacker's reward  is", V_attack[mdp.init])
    initial = mdp.states.index(mdp.init) # deterministic initial state.
    moment0=0
    moment01=0
    moment00=0
    small_epsilon = 1e-8
    print("The optimal value under the original MDP is", V_original[initial])
    for episode in range(episodes):
        V0 = policyEval(mdp, pol0)
        if V0[mdp.states.index(mdp.init)] < lb: # performance is worse than lower bound, constraint is violated.
            # gradient ascent for reward 0
            if episode % 100 == 0:
                print("The constraint is violated: V0 is", V0[mdp.states.index(mdp.init)])
            samples = mdp.generate_samples(pol0, sample_size)
            GradientCal0 = GradientCal(mdp, pol0, tau, mdp.reward)
            grad_pol0 = GradientCal0.dJ_dtheta(samples) # gradient ascent one step in the original MDP.
            if np.linalg.norm(grad_pol0, ord=np.inf) < tolerance:  # Stop if gradient is too small
                break
            moment0 += grad_pol0 ** 2
            theta0 +=  lr * grad_pol0 / (np.sqrt(moment0) + small_epsilon)    # Update step
            pol0 = Policy(mdp.states, mdp.actlist, False, theta_to_policy(mdp, theta0, tau))
        else:
            if episode % 100 == 0:
                print("The constraint is satisfied: V0 is", V0[mdp.states.index(mdp.init)])
            marg_MDP0 = marginalizedMDP(augmdp, pol1, 1) # marginalize out the policy for trigger
            aug_adv_reward0, aug_policy0 = covert_reward_pol_augmdp_pol0(augmdp, pol0, adv_reward) # compute the policy for player 1 in the augmented state space
            samples0 = marg_MDP0.generate_samples(aug_policy0, sample_size) # obtain samples from the augmented MDP.
            GradientCal1_0 = GradientCalTrigger.GradientCalTrigger(marg_MDP0, 0, pol0,  tau, adv_reward)
            grad_pol0_1 = GradientCal1_0.dJ_dtheta(samples0)  # gradient ascent one step in the marginalized MDP with policy 1.
            moment00 += grad_pol0_1 ** 2
            theta0 += lr * grad_pol0_1 / (np.sqrt(moment00) + small_epsilon)  # Update step
            pol0 = Policy(mdp.states, mdp.actlist, False, theta_to_policy(mdp, theta0, tau))
            # compute the gradient for policy 1
            marg_MDP1 = marginalizedMDP(augmdp, pol0, 0)  # marginalize out the policy for the backdoor policy
            aug_adv_reward1, aug_policy1 = covert_reward_pol_augmdp(augmdp,  pol1, adv_reward)
            samples1 = marg_MDP1.generate_samples(aug_policy1, sample_size) # obtain samples from the augmented MDP.
            GradientCal1_1 = GradientCalTrigger.GradientCalTrigger(marg_MDP1, 1, pol1, tau, adv_reward)
            grad_pol1_1 = GradientCal1_1.dJ_dtheta(samples1)  # gradient ascent one step in the marginalized MDP with policy 1.

            moment01 += grad_pol1_1 ** 2
            theta1 += lr * grad_pol1_1 / (np.sqrt(moment01) + small_epsilon)  # Update step
            # theta1 += learning_rate2 * grad_pol1_1  # Update step
            if np.linalg.norm(grad_pol0_1, ord=np.inf) < tolerance and np.linalg.norm(grad_pol1_1, ord=np.inf) < tolerance:  # Stop if gradient is too small
                break
            V1 = policyEval(augmdp,  get_joint_policy(augmdp, pol0, pol1))
            V1_iter.append(V1[augmdp.states.index(augmdp.init)])
            print("The value of the attacker's MDP under trigger:", V1[augmdp.states.index(augmdp.init)])
            #input("Press Enter to continue...")
            V0 =  policyEval(mdp,  pol0)
            V0_iter.append(V0[mdp.states.index(mdp.init)])
            print("The value for the original MDP:", V0[mdp.states.index(mdp.init)])
            #   if episode % 100 == 0:
            #       print(f"Episode {episode}: ")


    with open(f"{path}/V0.pkl", "wb") as file_v0:  # "rb" means read in binary mode
        pickle.dump(V0_iter, file_v0)
    with open(f"{path}/V1.pkl", "wb") as file_v1:  # "rb" means read in binary mode
        pickle.dump(V1_iter, file_v1)
    with open(f"{path}/pol0.pkl", "wb") as file_pol0:  # "rb" means read in binary mode
        pickle.dump(pol0, file_pol0)
    with open(f"{path}/pol1.pkl", "wb") as file_pol1:  # "rb" means read in binary mode
        pickle.dump(pol1, file_pol1)
    with open(f"{path}/v_original.pkl", "wb") as file_v0:  # "rb" means read in binary mode
        pickle.dump(V_original, file_v0)
    with open(f"{path}/pol0_opt.pkl", "wb") as file_all:  # "rb" means read in binary mode
        pickle.dump(pol0_opt, file_all)
    plot_results(path, V0_iter, V1_iter, V0_iter_attack, episodes, lb)
    return



def switchingGradient_no_marginalization(mdp,epsilon, adv_reward, trigger, augmdp,  K, path,  episodes=1000, lr=0.01,   tolerance=1e-2):
    """

    :param mdp: original MDP
    :param epsilon: performance degradtion suboptimality
    :param adv_reward: reward function r_b
    :param trigger: trigger policy class
    :param augmdp: augmented MDP constructed from MDP and a set of transition functions.
    :param K: number of transition functions for the trigger
    :param path: directory for result.
    :param episodes: upper bound on the number of episode
    :param lr: used in adam update
    :param tolerance: stopping criteria threshold.
    :return: pi0, pi1

    """
    original_states = mdp.states
    original_actlist = mdp.actlist
    trigger_states = trigger.states
    trigger_acts = trigger.actlist
    pol0 = Policy(original_states, original_actlist, False) # randomized policy, initialized to a uniform random one.
    pol1 = Policy(trigger_states, trigger_acts, False)
    theta0 = policy_to_theta(mdp, pol0)
    lb= get_lowerbound(mdp, degrade_percent =epsilon)
    tau = 0.1 # temperature
    theta1 = np.zeros(len(trigger.states)*K)
    V0_iter= []
    V1_iter= []
    V0_iter_attack = []
    sample_size = 10
    V_original, pol0_opt =   valueIter(mdp, tau)
    V_attack  = policyEval(mdp,  pol0_opt, 0.01, adv_reward)
    V_0 = policyEval(mdp,  pol0_opt, 0.01)
    print("The value of optimal policy for attacker's reward  is", V_attack[mdp.states.index(mdp.init)])
    initial = mdp.states.index(mdp.init) # deterministic initial state.
    moment0=0
    moment01=0
    moment00=0
    small_epsilon = 1e-8 # preventing devision by 0.
    print("The optimal value under the original MDP is", V_original[initial])
    for episode in range(episodes):
        # lr = lr * math.exp(-0.0004 * episode)
        pol0 = Policy(mdp.states, mdp.actlist, False, theta_to_policy(mdp, theta0, tau))
        pol1 = Policy(trigger.states, trigger.actlist, False, theta_to_policy(trigger, theta1, tau))

        joint_policy = get_joint_policy(augmdp, pol0, pol1)
        V0 = policyEval(mdp, pol0)
        joint_policy = get_joint_policy(augmdp, pol0, pol1)
        V1 = policyEval(augmdp, joint_policy, 0.01)
        V0_underattack = policyEval(augmdp,  joint_policy, 0.01, augmdp.agent_reward)
        V0_iter.append(V0[mdp.states.index(mdp.init)])
        V1_iter.append(V1[augmdp.states.index(augmdp.init)])
        V0_iter_attack.append(V0_underattack[augmdp.states.index(augmdp.init)])
        if V0[mdp.states.index(mdp.init)] < lb: # performance is worse than lower bound, constraint is violated.
            # gradient ascent for reward 0
            if episode % 100 == 0:
                print("The constraint is violated: V0 is", V0[mdp.states.index(mdp.init)])
            samples,_ = mdp.generate_samples(pol0, sample_size)
            GradientCal0 = GradientCal(mdp, pol0, tau, mdp.reward)
            grad_pol0 = GradientCal0.dJ_dtheta(samples) # gradient ascent one step in the original MDP.
            moment0 += grad_pol0 ** 2
            theta0 +=  lr * grad_pol0 / (np.sqrt(moment0) + small_epsilon)    # Update step
        else:
            if episode% 100 == 0:
                print("The constraint is satisfied: V0 is", V0[mdp.states.index(mdp.init)])
            samples, obs_samples = augmdp.generate_samples(joint_policy, sample_size) #TODO: ADD OBSERVED SAMPLES.
            GradientCal1_0 = GradientCalTrigger.GradientCalTrigger(augmdp, 0, pol0,  tau, adv_reward)
            grad_pol0_1 = GradientCal1_0.dJ_dtheta(samples)  # gradient ascent one step in the marginalized MDP with policy 1.
            moment00 += grad_pol0_1 ** 2
            theta0 += lr * grad_pol0_1 / (np.sqrt(moment00) + small_epsilon)  # Update step
            # compute the gradient for policy 1
            GradientCal1_1 = GradientCalTrigger.GradientCalTrigger(augmdp, 1, pol1, tau, adv_reward)

            grad_pol1_1 = GradientCal1_1.dJ_dtheta(obs_samples)  # TODO: observed_samples. # gradient ascent one step in the marginalized MDP with policy 1.
            moment01 += grad_pol1_1 ** 2
            theta1 += lr * grad_pol1_1 / (np.sqrt(moment01) + small_epsilon)  # Update step
            # theta1 += learning_rate2 * grad_pol1_1  # Update step
            #if np.linalg.norm(grad_pol0_1, ord=np.inf) < tolerance and np.linalg.norm(grad_pol1_1, ord=np.inf) < tolerance:  # Stop if gradient is too small
            #    break
            #V1 = policyEval(augmdp,  get_joint_policy(augmdp, pol0, pol1))
            #V1_iter.append(V1[augmdp.states.index(augmdp.init)])
            #print("The value of the attacker's MDP under trigger:", V1[augmdp.states.index(augmdp.init)])
            #input("Press Enter to continue...")
            #V0 =  policyEval(mdp,  pol0)
            #V0_iter.append(V0[mdp.states.index(mdp.init)])
            #print("The value for the original MDP:", V0[mdp.states.index(mdp.init)])
            #   if episode % 100 == 0:
            #       print(f"Episode {episode}: ")


    with open(f"{path}/V0.pkl", "wb") as file_v0:  # "rb" means read in binary mode
        pickle.dump(V0_iter, file_v0)
    with open(f"{path}/V1.pkl", "wb") as file_v1:  # "rb" means read in binary mode
        pickle.dump(V1_iter, file_v1)
    with open(f"{path}/V0_attack.pkl", "wb") as file_v0_attack:  # "rb" means read in binary mode
        pickle.dump(V0_iter_attack, file_v0_attack)
    with open(f"{path}/pol0.pkl", "wb") as file_pol0:  # "rb" means read in binary mode
        pickle.dump(pol0, file_pol0)
    with open(f"{path}/pol1.pkl", "wb") as file_pol1:  # "rb" means read in binary mode
        pickle.dump(pol1, file_pol1)
    with open(f"{path}/v_original.pkl", "wb") as file_v0:  # "rb" means read in binary mode
        pickle.dump(V_original, file_v0)
    with open(f"{path}/pol0_opt.pkl", "wb") as file_all:  # "rb" means read in binary mode
        pickle.dump(pol0_opt, file_all)
    plot_results(path, V0_iter, V1_iter, V0_iter_attack, episodes, lb)
    return pol0, pol1

def batch_test_switchingGradient(mdp, adv_reward, trigger, augmdp, K, base_path, episodes=1000, lr=0.01, tolerance=1e-2):
    """
    Run switchingGradient_no_marginalization for a batch of epsilon values and save results in separate folders.
    After running, compare the final V0_iter[-1] and V1_iter[-1] for each epsilon and return two lists.
    """
    import os
    import pickle
    epsilons = [0.1, 0.15, 0.2, 0.25]
    V0_final = []
    V1_final = []
    V0_final_attack = []
    for eps in epsilons:
        path = os.path.join(base_path, f"epsilon_{eps}")
        os.makedirs(path, exist_ok=True)
        print(f"Running for epsilon={eps}, results will be saved in {path}")
        switchingGradient_no_marginalization(
            mdp, eps, adv_reward, trigger, augmdp, K, path,
            episodes=episodes, lr=lr, tolerance=tolerance
        )
        # Load results
        try:
            with open(os.path.join(path, "V0.pkl"), "rb") as f0:
                V0_iter = pickle.load(f0)
            with open(os.path.join(path, "V1.pkl"), "rb") as f1:
                V1_iter = pickle.load(f1)
            with open(os.path.join(path, "V0_iter_attack.pkl"), "rb") as f2:
                V0_iter_attack = pickle.load(f2)
            V0_final.append(V0_iter[-1] if len(V0_iter) > 0 else None)
            V1_final.append(V1_iter[-1] if len(V1_iter) > 0 else None)
            V0_final_attack.append(V0_iter_attack[-1] if len(V0_iter_attack) > 0 else None)
        except Exception as e:
            print(f"Error loading results for epsilon={eps}: {e}")
            V0_final.append(None)
            V1_final.append(None)
            V0_final_attack.append(None)
    return V0_final, V1_final, V0_final_attack

def plot_final_vs_epsilon(epsilons, V0_final, V1_final, V0_final_attack, path="./results"):
    """
    Plot V0_final, V1_final, and V0_final_attack versus epsilons in three subplots.
    Also print the last value of V0_final_attack and save the figure to the specified path.
    """
    import os
    import matplotlib.pyplot as plt
    os.makedirs(path, exist_ok=True)
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    # V0_final vs epsilon (first)
    axs[0].plot(epsilons, V0_final, marker='o', label='V0_final')
    axs[0].set_xlabel('Epsilon')
    axs[0].set_ylabel('Final V0')
    axs[0].set_title('Final V0 vs Epsilon')
    axs[0].legend()
    axs[0].grid(True)
    # V0_final_attack vs epsilon (second)
    axs[1].plot(epsilons, V0_final_attack, marker='o', color='green', label='V0_final_attack')
    axs[1].set_xlabel('Epsilon')
    axs[1].set_ylabel('Final V0 under attack')
    axs[1].set_title('Final V0 under attack vs Epsilon')
    axs[1].legend()
    axs[1].grid(True)
    # V1_final vs epsilon (third)
    axs[2].plot(epsilons, V1_final, marker='o', color='orange', label='V1_final')
    axs[2].set_xlabel('Epsilon')
    axs[2].set_ylabel('Final V1')
    axs[2].set_title('Final V1 vs Epsilon')
    axs[2].legend()
    axs[2].grid(True)
    plt.tight_layout()
    fig.savefig(os.path.join(path, "final_vs_epsilon.png"))
    plt.show()
    print(f"Last V0_final_attack: {V0_final_attack[-1]}")

def plot_v0_and_attack_vs_epsilon(epsilons, V0_final, V0_final_attack, path="./results"):
    import os
    import matplotlib.pyplot as plt
    os.makedirs(path, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(epsilons, V0_final, marker='o', label='V0_final')
    plt.plot(epsilons, V0_final_attack, marker='o', color='green', label='V0_final_attack')
    plt.xlabel('Epsilon')
    plt.ylabel('Final Value')
    plt.title('Final V0 and V0 under Attack vs Epsilon')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(path, "v0_and_attack_vs_epsilon.png"))
    plt.show()