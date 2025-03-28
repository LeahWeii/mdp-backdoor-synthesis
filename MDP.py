
__author__ = "Jie Fu"
__date__ = "23 August 2024"



from scipy import stats
import numpy as np
import random
from pydot import Dot, Edge, Node
import copy



class MDP:
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

    def __init__(self, init=None, actlist=[], states=[], prob=dict([]), trans=dict([]), reward = dict([]), gamma=0.9):
        self.init = init
        self.actlist = actlist
        self.states = states
        self.prob = prob
        self.trans = trans #alternative for prob
        self.suppDict = dict([])
        if reward == dict([]):
            self.reward = {s:{a:0 for a in actlist} for s in states}
        else:
            self.reward = reward
        self.act_len = len(actlist)
        self.theta_size = len(actlist) * len(states)
        self.gamma = gamma


    def get_init_vec(self):
        if type(self.init) == np.array:
            self.init_vec = self.init
        else:
            temp = np.zeros(len(self.states), dtype=int)
            index = self.states.index(self.init)
            temp[index] = 1
            self.init_vec = temp
        return

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
        self.prob[action][i,j] = p
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
                self.suppDict[(s,a)] = self.supp(s,a)
        return

    def supp(self, state, action):
        """
        :param state:
        :param action:
        :return: a set of next states that can be reached with nonzero probability
        """
        supp = set([])
        for next_state in self.states:
            if self.P(state,action, next_state) !=0:
                supp.add(next_state)
        return supp

    def get_prec(self,state, act):
        # given a state and action, compute the set of states from which by taking that action, can reach that state with a nonzero probability.
        prec = set([])
        for pre_state in self.states:
            if self.P(pre_state, act, state) > 0:
                prec.add(pre_state)
        return prec

    def get_prec_anyact(self,state):
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



    def show_diagram(self, path='./graph.png'):  # pragma: no cover
        """
            Creates the graph associated with this MDP
        """
        # Nodes are set of states

        graph = Dot(graph_type='digraph', rankdir='LR')
        nodes = {}
        for state in self.states:
            if True:#tstate == self.init:
                pass
                # color start state with green
                # initial_state_node = Node(
                #        str(state),
                #         style='filled',
                #         peripheries=2,
                #         fillcolor='#66cc33')
                # nodes[str(state)] = initial_state_node
                # graph.add_node(initial_state_node)
            else:
                state_node = Node(str(state))
                nodes[str(state)] = state_node
                graph.add_node(state_node)
        # adding edges
        for state in self.states:
            i = self.states.index(state)
            for act in self.actlist:
                for next_state in self.states:
                    j  = self.states.index(next_state)
                    if self.prob[act][i,j] != 0:
                        weight = self.prob[act][i,j]
                        graph.add_edge(Edge(
                            nodes[str(state)],
                            nodes[str(next_state)],
                            label = act + str(': ') + str(weight)
                        ))
        if path:
            graph.write_png(path)
        return graph


    #policy gradient, in the algorithm define two mdps: M0 with R0 and Mp with Rp
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

    #generate samples

    def step(self, state, action):
        """Simulate a transition given a state and action."""
        next_state = random.choices(
            self.states, weights= self.prob[action][self.states.index(state), :], k=1)[0]
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
        traj = []
        self.get_init_vec()
        st= random.choices(self.states, weights=self.init_vec, k=1)[0]
        for _ in range(max_steps):
            # st_index = self.states.index(st)
            act = random.choices(self.actlist, weights=policy.policy[st], k=1)[0]
            next_state, step_reward = self.step(st, act)
            traj.append((st, act, next_state, step_reward))
            st = next_state
        return traj

    def generate_samples(self, policy, max_num = 10, max_steps=10):
        samples = []
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

        if policy==None:
            if deterministic:
                # Deterministic policy maps each state to a single action
                self.policy = {state: np.random.choice(self.actlist) for state in self.states}
            else:
                # Stochastic policy assigns a probability distribution over actions for each state
                self.policy ={state: np.ones(len(self.actlist)) / len(self.actlist) for state in self.states}
                     # uniform distribution
        else:
            self.policy = policy

    def get_action(self, state):
        """Returns an action based on the current policy."""
        if self.deterministic:
            return self.policy[state]
        else:
            pvec =self.policy[state]
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
        self.policy[state]= pvec
        return