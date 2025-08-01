# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 12:47:47 2023

@author: hma2
"""
import numpy as np

class GradientCal:
    def __init__(self, mdp, policy,   tau=1,  reward= None):
        self.mdp = mdp
        if reward == None:
            reward = mdp.reward
        self.st_len = len(mdp.states)
        self.act_len = len(mdp.actlist)
        self.x_size = self.st_len * self.act_len
        self.tau = tau
        self.policy = policy   #self.theta is the softmax parameterization of the policy.
        #self.P_matrix = self.construct_P()
        #self.epsilon = epsilon
        #0 is not using approximate_policy, 1 is using approximate_policy


    def dJ_dtheta(self, trajlist):
        # grdient of value function respect to theta
        # sample based method
        # returns dJ_dtheta_i, 1*NM matrix
        N = len(trajlist) # the total number of trajectories
        grad = 0
        for rho in trajlist:
            # print("trajectory is:", rho)
            grad += self.drho_dtheta(rho) * self.mdp.reward_traj(rho)
            # print(self.drho_dtheta(rho))
        # print("grad is:", grad)
        return 1 / N * grad

    
    def drho_dtheta(self, rho):
        if len(rho) == 1:
            return np.zeros(self.x_size)
        st = rho[0][0]
        act = rho[0][1]
        rho = rho[1:]
        return self.dPi_dtheta(st, act) + self.drho_dtheta(rho)
    
    def dPi_dtheta(self, st, act):
        #dlog(pi)_dtheta
        grad = np.zeros(self.x_size)
        st_index = self.mdp.states.index(st)
        act_index = self.mdp.actlist.index(act)
        Pi = self.policy.policy[st]
        # print("Pi:", Pi)
        for i in range(self.act_len):
            if i == act_index:
                grad[st_index * self.act_len + i] = 1/self.tau * (1.0 - Pi[i])
            else:
                grad[st_index * self.act_len + i] = 1/self.tau * (0.0 - Pi[i])
        #grad is a vector x_size * 1
        return grad

