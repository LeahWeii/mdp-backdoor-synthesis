
import random
import numpy as np
import string
import MDP
import pickle
from FSCTrigger import FSCTrigger
import backdoorSolver_Adam

with open("ex2/mdp1.pkl", "rb") as file1:  # "rb" means read in binary mode
    mdp1  = pickle.load(file1)

k= 3 # to change for example
with open("ex2/sampledtrans.pkl", "rb") as file2:  # "rb" means read in binary mode
    sampledTrans  = pickle.load(file2)
mdp1.getRewardMatrix()

max_reward = round(np.max(mdp1.reward_matrix))
adv_reward = {s: {a: max_reward- mdp1.reward[s][a] for a in mdp1.actlist} for s in mdp1.states}
sampledMDPs = []
states = mdp1.states.copy()
acts = mdp1.actlist.copy()
init = 0
count = 1
for prob in sampledTrans:
    mdp_temp = MDP.MDP(mdp1.init, acts, states)
    mdp_temp.prob = prob
    mdp_temp.show_diagram(f"ex2/figs/mdp_{count}_dot.dot", f"ex2/figs/mdp_{count}_graph.png")
    sampledMDPs.append(mdp_temp)
    count += 1
mdp1.show_diagram('ex2/figs/original_mdp_dot.png', 'ex2/figs/original_mdp_graph.png')

trigger = FSCTrigger(mdp1, k, 2)
# constructing the transition function of the trigger.

augmdp = backdoorSolver_Adam.get_augMDP(mdp1, trigger, sampledMDPs, adv_reward)
# This augmented MDP has very sparse transition matrix. should use sparse matrix for future.
# backdoorSolver_Adam.switchingGradient(mdp1, adv_reward, trigger, augmdp, k)


# warm-starting part
epsilon = 0.05
backdoorSolver_Adam.switchingGradient_no_marginalization(mdp1, epsilon, adv_reward, trigger, augmdp, k, './ex_memory_4')

print("complete ...")
