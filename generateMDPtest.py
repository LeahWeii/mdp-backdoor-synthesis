
from MDPgenerator import *

# this code generates the original MDP and a set of perturbed MDPs and save it. Use for ex-small.py
mdp1= generateMDP(5,2) # general a small MDP with 5 states and 2 actions randomly.

with open("mdp1.pkl", "wb") as file1:  # "wb" means write in binary mode
    pickle.dump(mdp1, file1)

epsilon = 0.05
K=2
sampledTrans = sample_k_trans(mdp1, K, epsilon) # return a list of perturbased transition matrix.

# Load from a file
with open("sampledtrans.pkl", "wb") as file2:  # "rb" means read in binary mode
    pickle.dump(sampledTrans, file2)
