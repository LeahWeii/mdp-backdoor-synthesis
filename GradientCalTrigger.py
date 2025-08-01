


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

    def dJ_dtheta_obs(self, trajlist):
        # gradient of value function respect to theta
        # sample based method
        # returns dJ_dtheta_i, 1*NM matrix
        N = len(trajlist)  # the total number of trajectories
        grad = 0
        for rho in trajlist:
            # print("trajectory is:", rho)
            grad += self.drho_dtheta_obs(rho) * self.mdp.reward_traj(rho)
            # print(self.drho_dtheta(rho))
        # print("grad is:", grad)
        return 1 / N * grad

    def drho_dtheta_obs(self, rho):
        if len(rho) == 1:
            return np.zeros(self.y_size)
        st = rho[0]
        act = rho[1]
        rho = rho[1:]
        # Handle partial observation: if state is None, gradient contribution is zero
        if st is None or st[self.player_id] is None:
            return self.drho_dtheta_obs(rho)
        return self.dPi_dtheta_obs(st, act) + self.drho_dtheta_obs(rho)

    def dPi_dtheta_obs(self, st, act):
        # dlog(pi)_dtheta
        grad = np.zeros(self.y_size)
        # Handle partial observation: if state is not observed, return zero gradient
        if st is None or st[self.player_id] is None:
            return grad
        # Check if the observed state is in the policy's state space
        if st[self.player_id] not in self.policy.states:
            # Handle unknown/unobserved states - could return zero or use a default policy
            print(self.player_id, st[self.player_id])
        st_index = self.policy.states.index(st[self.player_id])
        act_index = self.policy.actlist.index(act[self.player_id])
        Pi = self.policy.policy[st[self.player_id]]
        act_len = len(self.policy.actlist)
        for i in range(act_len):
            if i == act_index:
                grad[st_index * act_len + i] = 1 / self.tau * (1.0 - Pi[i])
            else:
                grad[st_index * act_len + i] = 1 / self.tau * (0.0 - Pi[i])
        return grad


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
        st_index = self.policy.states.index(st) # the local state index and action index for the player i'
        act_index = self.policy.actlist.index(act) # change.
        Pi = self.policy.policy[st]
        # print("Pi:", Pi)
        act_len = len(self.policy.actlist)
        for i in range(act_len):
            if i == act_index:
                grad[st_index * act_len + i] = 1 / self.tau * (1.0 - Pi[i])
            else:
                grad[st_index * act_len + i] = 1 / self.tau * (0.0 - Pi[i])
        # grad is a vector y_size * 1
        return grad


