
import random
import numpy as np
import string
from MDP import *
import pickle
from FSCTrigger import FSCTrigger
with open("mdp1.pkl", "rb") as file1:  # "rb" means read in binary mode
    mdp1  = pickle.load(file1)


with open("sampledtrans.pkl", "rb") as file2:  # "rb" means read in binary mode
    sampledTrans  = pickle.load(file2)
trigger = FSCTrigger(mdp1, 2)
# constructing the transition function of the trigger.
