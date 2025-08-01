
import numpy as np
import copy
import random
from POMDP import MDPwEmission
class GridWorld(MDPwEmission):
    def __init__(self, width, height, stoPar, init, F, G, obstacles, Barrier, gamma, tau):
        self.width = width
        self.height = height
        self.stoPar = stoPar
        self.actions = {'E':(1,0), 'W':(-1, 0), 'S': (0, -1), 'N':(0, 1)} #E, W, S, N, T for STOP
        self.actlist= list(self.actions.keys())
        self.complementA = self.getComplementA()
        self.states = self.getstate()
        self.observations= copy.deepcopy(self.states)
        self.observations.append(-1)
        self.addBarrier(Barrier)
        self.F = F
        self.G = G
        self.p_obs = 0.8
        self.obstacles = obstacles
        self.init = init
        self.trans  = self.gettrans()
        self.get_emission_matrix()
        self.get_prob()
        self.reward  = self.leader_reward()
        self.gamma = gamma
        self.tau = tau


    def is_terminal(self, state):
        if state in self.F or state in self.G or state == (-1, -1) :
            return True
        else:
            return False

    def getstate(self):
        states = []
        for i in range(self.width):
            for j in range(self.height):
                states.append((i, j))
        states.append((-1,-1)) # not needed for infinite horizon.
        return states

    def checkinside(self, st):
        if st in self.states:
            return True
        return False

    def getComplementA(self):
        complementA = {}
        complementA[(0, 1)] = [(1, 0), (-1, 0)]
        complementA[(0, -1)] = [(1, 0), (-1, 0)]
        complementA[(1, 0)] = [(0, 1), (0, -1)]
        complementA[(-1, 0)] = [(0, 1), (0, -1)]
        return complementA



    def get_transitions(self, state, action):
        stoPar = self.stoPar
        trans = {}
        for st in self.states:
            trans[st] = {}
        if self.is_terminal(st):
            for act in self.actions:
                trans[st][act] = {ns:0 for ns in self.states}
                trans[st][act][(-1,-1)] = 1.0
        else:
            x, y = state
            next_x, next_y = x, y
            for act in self.actlist:
                trans[st][act] = {}
                if act == "N":
                    next_y = min(y + 1, self.height - 1)
                elif act == "S":
                    next_y = max(y - 1, 0)
                elif act == "W":
                    next_x = max(x - 1, 0)
                elif act == "E":
                    next_x = min(x + 1, self.width - 1)
                next_state = (next_x, next_y)
                trans[st][act][next_state] += 1- 2*stoPar

        # Check if next state is an obstacle
        if next_state in self.obstacles:
            next_state = state
    def gettrans(self):
        #Calculate transition
        stoPar = self.stoPar
        trans = {}
        for st in self.states:
            trans[st] = {}
            if not self.is_terminal(st):
                for act in self.actlist:
                    trans[st][act] = {ns:0 for ns in self.states}
                    trans[st][act][st] = 0
                    tempst = tuple(np.array(st) + np.array(self.actions[act]))
                    if self.checkinside(tempst):
                        trans[st][act][tempst] = 1 - 2*stoPar
                    else:
                        trans[st][act][st] += 1- 2*stoPar
                    for act_ in self.complementA[self.actions[act]]:
                        tempst_ = tuple(np.array(st) + np.array(act_))
                        if self.checkinside(tempst_):
                            trans[st][act][tempst_] = stoPar
                        else:
                            trans[st][act][st] += stoPar
            else:
                for act in self.actions:
                    trans[st][act] ={ns:0 for ns in self.states}
                    trans[st][act][(-1,-1)] = 1.0

        if self.checktrans(trans):
            return trans
        else:
            print("Transition is incorrect")

    def get_prob(self):
        n_states = len(self.states)
        self.prob = {a: np.zeros((n_states, n_states)) for a in self.actlist}
        for a in self.actlist:
            for i in range(n_states):
                st= self.states[i]
                for j in range(n_states):
                    ns = self.states[j]
                    self.prob[a][i,j]  = self.trans[st][a][ns]
        return



    def get_emission_matrix(self):
        """Get the emission matrix, which is a square matrix with rows as states and columns as observations.
        The value is the probability of observing the observation given the state."""
        self.emit_matrix = np.zeros((len(self.states), len(self.observations)))
        null_obs_idx = self.observations.index(-1) if -1 in self.observations else None
        for s in self.states:
            for o in self.observations:
                s_idx = self.states.index(s)
                o_idx = self.observations.index(o)
                if s == o:
                    self.emit_matrix[s_idx, o_idx] = self.p_obs
                    self.emit_matrix[s_idx, null_obs_idx] = 1 - self.p_obs if null_obs_idx is not None else 0
        return

    def get_observation(self, state):
        """Given a state, return an observation with probability p_obs."""
        if random.random() < self.p_obs:
            return state
        else:
            return -1 # -1 means no observation is made.

    def get_obs_supp(self, state):
        """Get the support of observations for a given state. This is a set of observations that can be observed from the state."""
        supp = set([])
        for o in self.observations:
            if self.get_emit_prob(state, o) != 0:
                supp.add(o)
        return supp

    def get_emit_prob(self, state, observation):
        s_idx = self.states.index(state)
        o_idx = self.observations.index(observation)
        return self.emit_matrix[s_idx, o_idx]


    def checktrans(self, trans):
        for st in self.states:
            for act in self.actlist:
                if abs(sum(trans[st][act].values())-1) > 0.01:
                    print("st is:", st, " act is:", act, " sum is:", sum(trans[st][act].values()))
                    return False
        return True


    def addBarrier(self, Barrierlist):
        #Add barriers in the world
        #If we want to add barriers, Add barriers first, then calculate trans, add True goal, add Fake goal, add IDS
        for st in Barrierlist:
            self.states.remove(st)

    def init_value(self):
        #Initial the value to be all 0
        return np.zeros(len(self.states))

    def update_reward(self, reward = []):
        #Update follower's reward
        if len(reward) >0:
            self.reward = {}
            i = 0
            for st in self.states:
                self.reward[st] = {}
                for act in self.actions:
                    self.reward[st][act] = reward[i]
                    i += 1
        else:
            self.initial_reward()

    def leader_reward(self):
        leader_reward = {}
        for st in self.states:
            leader_reward[st] = {}
            if st in self.F:
                for act in self.actions:
                    leader_reward[st][act] = 2.0
            elif st in self.G:
                for act in self.actions:
                    leader_reward[st][act] = 20.0
            else:
                for act in self.actions:
                    leader_reward[st][act] = 0.0
        return leader_reward

    def getcore(self, V, st, act):
        core = 0
        for st_, pro in self.trans[st][act].items():
            if st_ != (-1, -1):
                core += pro * V[self.states.index(st_)]
        return core

    def initial_reward(self):
        self.reward = {}
        for st in self.states:
            self.reward[st] = {}
            if st in self.G:
                for act in self.actions:
                    self.reward[st][act] = 10.0
            elif st in self.obstacles:
                for act in self.actions:
                    self.reward[st][act] = -5.0
            elif st in self.F:
                for act in self.actions:
                    self.reward[st][act] = 8.0
            else:
                for act in self.actions:
                    self.reward[st][act] = 0


    def policy_evaluation(self, reward, flag, policy):
        threshold = 0.00001
        if flag == 0:
            reward = self.reward_l
        else:
            self.update_reward(reward)
            reward = self.reward
        V = self.init_value()
        delta = np.inf
        while delta > threshold:
            V1 = V.copy()
            for st in self.states:
                temp = 0
                for act in self.actions:
                    if act in policy[st].keys():
                        temp += policy[st][act] * (reward[st][act] + self.gamma * self.getcore(V1, st, act))
                V[self.states.index(st)] = temp
            delta = np.max(abs(V-V1))
        return V