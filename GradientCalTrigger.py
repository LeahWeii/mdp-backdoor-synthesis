


def pi_theta(m, sa, theta):
    """
    :param m: the index of a finite sequence of observation, corresponding to K-step memory
    :param sa: the sensing action to be given
    :param theta: the policy parameter, the memory_size * sensing_action_size
    :return: the Gibbs policy given the finite memory
    """
    e_x = np.exp(theta[m, :] - np.max(theta[m, :]))
    return (e_x / e_x.sum(axis=0))[sa]


def log_policy_gradient(m, sa, theta):
    # A memory space for K-step memory policy
    memory_space = fsc.memory_space
    memory_size = fsc.memory_size
    gradient = np.zeros([memory_size, env.sensing_actions_size])
    memory = memory_space[m]
    senAct = env.sensing_actions[sa]
    for m_prime in range(memory_size):
        for a_prime in range(env.sensing_actions_size):
            memory_p = memory_space[m_prime]
            senAct_p = env.sensing_actions[a_prime]
            indicator_m = 0
            indicator_a = 0
            if memory == memory_p:
                indicator_m = 1
            if senAct == senAct_p:
                indicator_a = 1
            partial_pi_theta = indicator_m * (indicator_a - pi_theta(m_prime, a_prime, theta))
            gradient[m_prime, a_prime] = partial_pi_theta
    return gradient


# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 12:47:47 2023

@author: hma2
"""

import numpy as np
import MDP
import math
import Sample
import pickle
from mdpSolver import *


class GradientCalTrigger:
    def __init__(self, mdp, player_id,  policy, tau=1, reward=None):
        self.mdp = mdp
        self.player_id = player_id
        if reward == None:
            reward = mdp.reward
        self.st_len = len(mdp.states)
        self.act_len = len(mdp.actlist) # the action set is the same for MDP and the policy
        self.x_size = self.st_len * self.act_len # this is the size of the MDP. not the policy
        self.tau = tau
        self.policy = policy  # self.theta is the softmax parameterization of the policy.
        self.y_size = len(self.policy.states)* len(self.policy.actlist)

        # self.P_matrix = self.construct_P()
        # self.epsilon = epsilon
        # 0 is not using approximate_policy, 1 is using approximate_policy

    def J_func(self, policy, epsilon):
        V = policyEval(self.mdp, policy, epsilon)
        J = self.mdp.get_init_vec().dot(V)
        return J, policy

    def dJ_dtheta(self, trajlist):
        # grdient of value function respect to theta
        # sample based method
        # returns dJ_dtheta_i, 1*NM matrix
        N = len(trajlist)  # the total number of trajectories
        grad = 0
        for rho in trajlist:
            # print("trajectory is:", rho)
            grad += self.drho_dtheta(rho) * self.mdp.reward_traj(rho)
            # print(self.drho_dtheta(rho))
        # print("grad is:", grad)
        return 1 / N * grad

    def construct_P(self):
        P = np.zeros((self.x_size, self.x_size))
        for i in range(self.st_len):
            for j in range(self.act_len):
                for next_st in self.mdp.states:
                    next_index = self.mdp.states.index(next_st)
                    P[i * self.act_len + j][next_index * self.act_len: (next_index + 1) * self.act_len] = \
                    self.mdp.prob[self.mdp.actlist[j]][i, next_index]
        return P

    def policy_to_theta(self, policy):
        policy_m = np.zeros(self.y_size)
        i = 0
        act_len = len(policy.actlist)
        for st in self.policy.states:
            for j in range(act_len):
                policy_m[i] = policy.policy[st][j]
                i += 1
        return policy_m

    def drho_dtheta(self, rho):
        if len(rho) == 1:
            return np.zeros(self.y_size)
        st = rho[0][0]
        act = rho[0][1]
        rho = rho[1:]
        return self.dPi_dtheta(st, act) + self.drho_dtheta(rho)

    def dPi_dtheta(self, st, act):
        # dlog(pi)_dtheta
        grad = np.zeros(self.y_size)
        st_index = self.policy.states.index(st[self.player_id]) # the local state index and action index for the player i'
        act_index = self.policy.actlist.index(act)
        Pi = self.policy.policy[st[self.player_id]]
        # print("Pi:", Pi)
        for i in range(self.act_len):
            if i == act_index:
                grad[st_index * self.act_len + i] = 1 / self.tau * (1.0 - Pi[i])
            else:
                grad[st_index * self.act_len + i] = 1 / self.tau * (0.0 - Pi[i])
        # grad is a vector y_size * 1
        return grad

    def update_policy(self, policy, N):
        # if self.approximate_flag:
        #     Leader uses approximate policy
        # policy = policy_convert(policy, self.mdp.actions)
        # self.sample.generate_traj(N, policy)
        # policy_ = self.sample.approx_policy()
        # self.policy_m = self.convert_policy(policy_)
        # self.policy = policy_convert(policy_, self.mdp.actions)
        # else:
        # Leader uses exact policy
        self.policy_m = self.convert_policy(policy)
        self.policy = policy_convert(policy, self.mdp.actions)
        self.sample.generate_traj(N, self.policy)

    def SGD(self, N):
        delta = np.inf
        J_old, policy = self.J_func()  # policy is exact policy
        policy_c = policy_convert(policy, self.mdp.actions)  # exact policy
        self.update_policy(policy, N)  # exact or approximate policy, depends on flag
        itcount = 1

        Jlist = []
        Jlist.append(J_old)
        while itcount <= 200:
            # while delta > self.epsilon:
            self.dJ_dx(N, policy_c)  #
            J_new, policy, V_att = self.J_func()  # exact policy
            policy_c = policy_convert(policy, self.mdp.actions)  # exact policy
            print("J_new:", J_new)
            Jlist.append(J_new)
            # update it to new policy
            self.update_policy(policy, N)
            delta = abs(J_new - J_old)
            print("delta:", delta)
            J_old = J_new
            # print(f"{itcount}th iteration")
            itcount += 1
            if itcount % 100 == 0:
                print(f'{itcount}th iteration, x is {self.x}')
        save_data(Jlist)
        print("Attacker's value is:", V_att)
        # print(policy)
        return self.x


def save_data(data):
    filename = "Jlist_modelfree200.pkl"
    file = open(filename, 'wb')
    pickle.dump(data, file)
    file.close()


def policy_convert(pi, action_list):
    # Convert a policy from pi[st][act] = pro to pi[st] = [pro1, pro2, ...]
    pi_list = {}
    for st in pi.keys():
        pro = []
        for act in action_list:
            pro.append(pi[st][act])
        pi_list[st] = pro
    return pi_list


def reward2list(reward, states, actions):
    reward_s = []
    for st in states:
        for act in actions:
            reward_s.append(reward[st][act])
    return np.array(reward_s)


def MDP_example():
    # This is the function used to generate small transition MDP example.
    mdp = MDP.create_mdp()
    V, policy = mdp.get_policy_entropy([], 1)
    # Learning rate influence the result from the convergence aspect. Small learning rate wll make the convergence criteria satisfy too early.
    lr_x = 0.02  # The learning rate of side-payment
    modifylist = [48]  # The action reward you can modify
    epsilon = 1e-6  # Convergence threshold
    weight = 0.5  # weight of the cost
    approximate_flag = 0  # Whether we use trajectory to approximate policy. 0 represents exact policy, 1 represents approximate policy
    GradientCal = GC(mdp, lr_x, policy, epsilon, modifylist, weight, approximate_flag)
    x_res = GradientCal.SGD(N=200)
    print(x_res)


def GridW_example():
    mdp = GW.createGridWorldBarrier_new2()
    V, policy = mdp.get_policy_entropy([], 1)
    lr_x = 0.01
    modifylist = [40, 116]
    epsilon = 1e-6
    weight = 0
    approximate_flag = 1
    GradientCal = GC(mdp, lr_x, policy, epsilon, modifylist, weight, approximate_flag)
    x_res = GradientCal.SGD(N=200)
    print(x_res)


def GridW_example_V2():
    mdp = GW2.createGridWorldBarrier_new2()
    V, policy = mdp.get_policy_entropy([], 1)
    lr_x = 0.02
    modifylist = [40]
    # modifylist = [15]
    epsilon = 1e-6
    weight = 0.3
    approximate_flag = 1
    GradientCal = GC(mdp, lr_x, policy, epsilon, modifylist, weight, approximate_flag)
    x_res = GradientCal.SGD(N=200)
    print(x_res)


if __name__ == "__main__":
    # MDP_example()
    GridW_example_V2()