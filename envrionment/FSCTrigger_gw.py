import itertools
import numpy as np
import ast
import re

class FSCTriggerGW:

    def __init__(self,mdp,  K=2):
        # The length of memory
        self.K = K
        self.actlist = list(range(K))
        self.init = None
        # The observations (the input of the finite state controller)
        self.get_memory_transition(mdp,K)




    def get_memory_transition(self,mdp, k):
        self.trans = {}
        self.states = [str([mdp.init])]
        self.init =  str([mdp.init])
        pointer = 0
        while pointer < len(self.states):
            memory_state = self.states[pointer]
            pointer += 1
            self.trans[memory_state] = {}
            if memory_state == 'l': # initialization
                for s in mdp.states:
                    new_m = str([s])
                    self.trans[memory_state][new_m] = new_m
                    if new_m not in self.states:
                        self.states.append(new_m)
            else:
                matches  = [ast.literal_eval(memory_state)]
                s =  matches[-1]  # the most recent state
                for a in mdp.actlist:
                    for ns in mdp.states:
                        if mdp.P(s,a,ns) !=0:
                            temp_m = " ".join([memory_state, str(ns)])
                            new_m = " ".join(temp_m.split()[-k:])
                            if new_m not in self.states:
                                self.states.append(new_m)
                            self.trans[memory_state][(s,a,ns)] = new_m
        return



class FSCTriggerGW_finite:

    def __init__(self,mdp,   K=2, memory_length=2):
        # The length of memory
        self.K = K
        self.actlist = list(range(K))
        self.init = None
        # The observations (the input of the finite state controller)
        self.get_memory_transition(mdp,    K, memory_length)

    def get_memory_transition(self,mdp , k, memory_length):
        self.trans = {}
        self.states = ['l']
        self.init = 'l'
        pointer = 0
        while pointer < len(self.states):
            memory_state = self.states[pointer]
            pointer += 1
            self.trans[memory_state] = {}
            if memory_state == 'l': # initialization
                for o in mdp.observations:
                    new_m = str([o])
                    self.trans[memory_state][o] = new_m
                    if new_m not in self.states:
                        self.states.append(new_m)

            else:
                # print("the memory state is: ", memory_state)
                matches  =   ast.literal_eval(memory_state)
                for no in mdp.observations:
                    temp_m = matches + [no]
                    print(temp_m)
                    if len(temp_m) > memory_length:
                        temp_m = temp_m[-memory_length:]
                    new_m = str(temp_m)
                    if new_m not in self.states:
                        self.states.append(new_m)
                    self.trans[memory_state][no] = new_m # removing action from the transition
        return