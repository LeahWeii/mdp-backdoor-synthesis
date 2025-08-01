__author__ = "Jie Fu"
__date__ = "23 August 2024"

from scipy import stats
import numpy as np
import random
from pydot import Dot, Edge, Node
import copy
import pickle
import ast


class MDPwEmission:
    """A Markov Decision Process, defined by an initial state,
        transition model --- the probability transition matrix, np.array
        prob[a][0,1] -- the probability of going from 0 to 1 with action a.
        and reward function. We also keep track of a gamma value, for
        use by algorithms. The transition model is represented
        somewhat differently from the text.  Instead of T(s, a, s')
        being probability number for each state/action/state triplet,
        we instead have T(s, a) return a list of (p, s') pairs.  We
        also keep track of the possible states, terminal states, and
        actlist for each state.  The input transitions is a
        dictionary: (state,action): list of next state and probability
        tuple.  AP: a set of atomic propositions. Each proposition is
        identified by an index between 0 -N.  L: the labeling
        function, implemented as a dictionary: state: a subset of AP."""

    def __init__(self, init=None, actlist=[], states=[], observations = [], prob=dict([]), trans=dict([]), reward=dict([]), p_obs=0.8, gamma=0.9):
        self.init = init
        self.actlist = actlist
        self.states = states
        self.observations = observations
        self.prob = prob
        self.p_obs = p_obs # simple observation with probability of observing the current state.
        self.emit = dict([])
        self.trans = trans  # alternative for prob
        self.suppDict = dict([])
        self.init_dist = None
        if reward == dict([]):
            self.reward = {s: {a: 0 for a in actlist} for s in states}
        else:
            self.reward = reward
        self.act_len = len(actlist)
        self.theta_size = len(actlist) * len(states)
        self.gamma = gamma
        self.get_emission_matrix()

    def get_init_vec(self):
        if type(self.init) == np.array:
            self.init_vec = self.init
        else:
            temp = np.zeros(len(self.states), dtype=int)
            index = self.states.index(self.init)
            temp[index] = 1
            self.init_vec = temp
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

    def getRewardMatrix(self):
        self.reward_matrix = np.zeros((len(self.states), len(self.actlist)))
        for s in self.states:
            for a in self.actlist:
                s_idx = self.states.index(s)
                act_idx = self.actlist.index(a)
                self.reward_matrix[s_idx, act_idx] = self.reward[s][a]
        return

    def T(self, state, action):
        """Transition model.  From a state and an action, return a row in the matrix for next-state probability."""
        i = self.states.index(state)
        return self.prob[action][i, :]

    def P(self, state, action, next_state):
        "Derived from the transition model. For a state, an action and the next_state, return the probability of this transition."
        i = self.states.index(state)
        j = self.states.index(next_state)
        return self.prob[action][i, j]

    def assign_P(self, state, action, next_state, p):
        i = self.states.index(state)
        j = self.states.index(next_state)
        self.prob[action][i, j] = p
        return

    def act(self, state):
        "Compute a set of enabled actlist from a given state"
        N = len(self.states)
        S = set([])
        for a in self.actlist:
            if not np.array_equal(self.T(state, a), np.zeros(N)):
                S.add(a)
        return S

    def labeling(self, s, A):
        """

        :param s: state
        :param A: set of atomic propositions
        :return: labeling function
        """
        self.L[s] = A

    def get_supp(self):
        """
        Compute a dictionary: (state,action) : possible next states.
        :return:
        """
        self.suppDict = dict([])
        for s in self.states:
            for a in self.actlist:
                self.suppDict[(s, a)] = self.supp(s, a)
        return

    def supp(self, state, action):
        """
        :param state:
        :param action:
        :return: a set of next states that can be reached with nonzero probability
        """
        supp = set([])
        for next_state in self.states:
            if self.P(state, action, next_state) != 0:
                supp.add(next_state)
        return supp

    def get_prec(self, state, act):
        # given a state and action, compute the set of states from which by taking that action, can reach that state with a nonzero probability.
        prec = set([])
        for pre_state in self.states:
            if self.P(pre_state, act, state) > 0:
                prec.add(pre_state)
        return prec

    def get_prec_anyact(self, state):
        # compute the set of states that can reach 'state' with some action.
        prec_all = set([])
        for act in self.actlist:
            prec_all = prec_all.union(self.get_prec(state, act))
        return prec_all

    def sample(self, state, action, num=1):
        """Sample the next state according to the current state, the action, and the transition probability. """
        if action not in self.act(state):
            return None  # Todo: considering adding the sink state
        N = len(self.states)
        i = self.states.index(state)
        next_index = np.random.choice(N, num, p=self.prob[action][i, :])[
            0]  # Note that only one element is chosen from the array, which is the output by random.choice
        return self.states[next_index]

    def show_diagram(self, dotpath='./dot_file.pkl', path='./graph.png'):  # pragma: no cover
        """
            Creates the graph associated with this MDP
        """
        # Nodes are set of states

        graph = Dot(graph_type='digraph', rankdir='LR')
        nodes = {}
        for state in self.states:
            if state == self.init:
                # color start state with green
                initial_state_node = Node(
                    str(state),
                    style='filled',
                    peripheries=2,
                    fillcolor='#66cc33')
                nodes[str(state)] = initial_state_node
                graph.add_node(initial_state_node)
            else:
                state_node = Node(str(state))
                nodes[str(state)] = state_node
                graph.add_node(state_node)
        # adding edges
        for state in self.states:
            i = self.states.index(state)
            for act in self.actlist:
                for next_state in self.states:
                    j = self.states.index(next_state)
                    if self.prob[act][i, j] != 0:
                        weight = np.round(self.prob[act][i, j], 2)
                        graph.add_edge(Edge(
                            nodes[str(state)],
                            nodes[str(next_state)],
                            label=act + str(': ') + str(weight)
                        ))
        if path:
            graph.write_png(path)

        with open(dotpath, "wb") as file1:  # "wb" means write in binary mode
            pickle.dump(self, file1)

        return graph

    # policy gradient, in the algorithm define two mdps: M0 with R0 and Mp with Rp
    def reward_traj(self, traj):
        if len(traj) > 1:
            r = traj[0][-1] + self.gamma * self.reward_traj(traj[1:])
        else:
            return traj[0][-1]
        return r

    def dJ_dtheta(self, Sample, policy):
        # grdient of value function respect to theta
        # sample based method
        # returns dJ_dtheta_i, 1*NM matrix
        N = len(Sample.trajlist)
        grad = 0
        for rho in Sample.trajlist:
            # print("trajectory is:", rho)
            grad += self.drho_dtheta(rho, policy) * self.reward_traj(rho, 0)
            # print(self.drho_dtheta(rho))
        # print("grad is:", grad)
        return 1 / N * grad

    def drho_dtheta(self, rho, policy):
        if len(rho) == 1:
            return np.zeros(self.theta_size)
        st = rho[0]
        act = rho[1]
        rho = rho[2:]
        return self.dPi_dtheta(st, act, policy) + self.drho_dtheta(rho, policy)

    def dPi_dtheta(self, st, act, policy):
        # dlog(pi)_dtheta
        grad = np.zeros(self.theta_size)
        st_index = self.states.index(st)
        act_index = self.actlist.index(act)
        Pi = policy[st]
        # print("Pi:", Pi)
        for i in range(self.act_len):
            if i == act_index:
                grad[st_index * self.act_len + i] = 1 / self.tau * (1.0 - Pi[i])
            else:
                grad[st_index * self.act_len + i] = 1 / self.tau * (0.0 - Pi[i])
        # grad is a vector x_size * 1
        return grad

    # generate samples

    def step(self, state, action):
        """Simulate a transition given a state and action."""
        if self.prob[action][self.states.index(state), :].sum() == 0:
            print('incorrect')
        next_state = random.choices(
            self.states, weights=self.prob[action][self.states.index(state), :], k=1)[0]
        reward = self.reward[state][action]
        return next_state, reward

    def step_2(self, state, action):
        """Simulate a transition given a state and action."""
        if self.prob[action][self.states.index(state), :].sum() == 0:
            print('incorrect')
        next_state = random.choices(
            self.states, weights=self.prob[action][self.states.index(state), :], k=1)[0]
        reward = self.reward[state][action]
        return next_state, reward

    def one_step_transition(self, st, act, st_lists, pro_lists):

        st_list = st_lists[st][act]
        pro_list = pro_lists[st][act]
        if all(x == 0 for x in pro_list):
            return None
        next_st = np.random.choice(len(st_list), 1, p=pro_list)[0]
        return st_list[next_st]

    def generate_sample(self, policy, max_steps=10):
        # pi here should be pi[st] = [pro1, pro2, ...]
        sys_traj = []
        self.get_init_vec()
        st = random.choices(self.states, weights=self.init_vec, k=1)[0]
        sys_st = st  # initialized the system states.
        for _ in range(max_steps):
            # st_index = self.states.index(st)
            act = random.choices(self.actlist, weights=policy.policy[sys_st], k=1)[0]
            next_state, step_reward = self.step(st, act)
            sys_next_state = next_state  # observation.
            sys_act = act
            sys_traj.append((sys_st, sys_act, sys_next_state, step_reward))
            sys_st = sys_next_state
        return sys_traj

    def generate_sample_2(self, policy, max_steps=10):
        # pi here should be pi[st] = [pro1, pro2, ...]
        sys_traj = []
        trigger_traj = []  #
        st = random.choices(self.states, weights=self.init_dist, k=1)[0]
        sys_st = st[0]  # initialized the system states.
        trigger_st = st[1]  # initialized the trigger states.
        for _ in range(max_steps):
            # st_index = self.states.index(st)
            act = random.choices(self.actlist, weights=policy.policy[st], k=1)[0]
            next_state, step_reward = self.step(st, act)
            sys_next_state = next_state[0]  # observation.
            sys_act = act[0]
            trigger_act = act[1]
            trigger_next_state = next_state[1]
            sys_traj.append((sys_st, sys_act, sys_next_state, step_reward))
            trigger_traj.append((trigger_st, trigger_act, trigger_next_state, step_reward))
            st = next_state
            sys_st = sys_next_state
            trigger_st = trigger_next_state  # observation. #TODO
        return sys_traj, trigger_traj

    def obs(self, state, p=0.8):
        """
        Observation function that modifies the first element and the corresponding
        last element in the list string (since they should be the same).

        Args:
            state

        Returns:
            Modified tuple where both the first element and last element in the list
            have probability p to remain the same and probability 1-p to become 'none'
        """
        # Parse the string to get the actual list

        # Apply observation noise to both the first element and last element in list
        observed_element = copy.deepcopy(state)
        if random.random() < p:
            pass
        else:
            observed_element[0] = 'None'
        return observed_element

    def generate_samples(self, policy, memory_length, max_num=10, max_steps=10):
        samples = []
        for _ in range(max_num):
            traj = self.generate_sample(policy, max_steps)
            samples.append(traj)
        return samples

    def generate_samples_2(self, policy, memory_length, max_num=10, max_steps=10):
        samples = []
        obs_samples = []
        for _ in range(max_num):
            traj, obs_traj = self.generate_sample_2(policy, max_steps)
            samples.append(traj)
            obs_samples.append(obs_traj)
        return samples, obs_samples

    def generate_samples_player0(self, policy, max_num=10, max_steps=10):
        samples = []
        obs_samples = []
        for _ in range(max_num):
            traj = self.generate_sample(policy, max_steps)
            samples.append(traj)
        return samples


class Policy:
    def __init__(self, states, actlist, deterministic=True, policy=None):
        """
        Initialize a policy for an MDP.

        :param state_space: List or range of states in the MDP.
        :param action_space: List or range of possible actions.
        :param deterministic: Boolean indicating whether the policy is deterministic or stochastic.
        """
        self.states = states
        self.actlist = actlist
        self.deterministic = deterministic

        if policy == None:
            if deterministic:
                # Deterministic policy maps each state to a single action
                self.policy = {state: np.random.choice(self.actlist) for state in self.states}
            else:
                # Stochastic policy assigns a probability distribution over actions for each state
                self.policy = {state: np.ones(len(self.actlist)) / len(self.actlist) for state in self.states}
                # uniform distribution
        else:
            self.policy = policy

    def get_action(self, state):
        """Returns an action based on the current policy."""
        if self.deterministic:
            return self.policy[state]
        else:
            pvec = self.policy[state]
            return np.random.choice(self.actlist, p=pvec)

    def update_policy(self, state, action, p):
        """
        Update the policy for a given state.

        :param state: The state to update the policy for.
        :param: If deterministic, a single action. If stochastic, a probability of an action
        """
        if self.deterministic:
            self.policy[state] = action  # Single action
        else:
            act_id = self.actlist.index(action)
            self.policy[state][act_id] = p

    def update_policy_actions(self, state, pvec):
        self.policy[state] = pvec
        return


import ast
import random

import ast
import random

import ast
import random


def convert_to_obs_traj(traj, k, p=0.8):
    """
    Convert a trajectory into observation trajectory with observation noise and finite memory.

    Args:
        traj: list of (((state, memory_str), action, (next_state, next_memory_str), reward))
        p: probability of observing the current state

    Returns:
        obs_traj: list of ((obs_state, obs_mem_str), action, s_next, reward)
    """
    obs_traj = []

    # 🔐 Safe check: find the first valid memory string to extract k
    # memory length

    obs_memory = []

    for ((state, _), action, s_next, reward) in traj:
        # Observe the current state with probability p
        obs = state if random.random() < p else 'None'

        # Update memory (rolling window)
        obs_memory.append(obs)
        if len(obs_memory) > k:
            obs_memory = obs_memory[-k:]

        # Build observation state
        obs_s_prev = (obs, str(obs_memory.copy()))

        obs_traj.append((obs_s_prev, action, s_next, reward))

    return obs_traj

