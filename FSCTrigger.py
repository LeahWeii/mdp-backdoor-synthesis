import itertools
import numpy as np



class FSCTrigger:

    def __init__(self,mdp,  K=2):
        # The length of memory
        self.K = K
        # The observations (the input of the finite state controller)
        self.get_memory_transition(mdp,K)


    def get_memory_transition(self,mdp, k):
        self.trans = {}
        self.memory_space = ['l']
        pointer = 0
        while pointer < len(self.memory_space):
            memory_state = self.memory_space[pointer]
            pointer += 1
            self.trans[memory_state] = {}
            if memory_state == 'l': # initialization
                for s in mdp.states:
                    new_m = str(s)
                    self.trans[memory_state][new_m] = new_m
                    if new_m not in self.memory_space:
                        self.memory_space.append(new_m)
            else:
                s = int(memory_state[-1]) # the most recent state
                for a in mdp.actlist:
                    for ns in mdp.states:
                        if mdp.P(s,a,ns) !=0:
                            temp_m = " ".join([memory_state, str(ns)])
                            new_m = " ".join(temp_m.split()[-k:])
                            if new_m not in self.memory_space:
                                self.memory_space.append(new_m)
                            self.trans[memory_state][(s,a,ns)] = new_m
        return
