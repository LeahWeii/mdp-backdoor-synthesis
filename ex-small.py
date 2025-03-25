
import random
import numpy as np
import string
import MDP
import pickle
from FSCTrigger import FSCTrigger
import backdoorSolver

with open("mdp1.pkl", "rb") as file1:  # "rb" means read in binary mode
    mdp1  = pickle.load(file1)

k= 2
with open("sampledtrans.pkl", "rb") as file2:  # "rb" means read in binary mode
    sampledTrans  = pickle.load(file2)
mdp1.getRewardMatrix()

max_reward = round(np.max(mdp1.reward_matrix))
adv_reward = {s: {a: max_reward- mdp1.reward[s][a] for a in mdp1.actlist} for s in mdp1.states}
sampledMDPs = []
states = mdp1.states.copy()
acts = mdp1.actlist.copy()
init = 0
for prob in sampledTrans:
    mdp_temp = MDP.MDP(mdp1.init, acts, states)
    mdp_temp.prob = prob
    sampledMDPs.append(mdp_temp)

trigger = FSCTrigger(mdp1, k)
# constructing the transition function of the trigger.

augmdp = backdoorSolver.get_augMDP(mdp1, trigger, sampledMDPs, adv_reward)
# This augmented MDP has very sparse transition matrix. should use sparse matrix for future.
backdoorSolver.switchingGradient(mdp1, adv_reward, trigger, augmdp, k)
print("complete ...")
