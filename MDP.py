
__author__ = "Jie Fu"
__date__ = "23 August 2024"



from scipy import stats
import numpy as np

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

    def __init__(self, init=None, actlist=[], states=[], prob=dict([]), trans=dict([]), reward = dict([])):
        self.init = init
        self.actlist = actlist
        self.states = states
        self.prob = prob
        self.trans = trans #alternative for prob
        self.suppDict = dict([])
        self.reward = dict([])
        self.act_len = len(actlist)
        self.theta_size = len(actlist) * len(states)

    def R(self, state):
        "Return a numeric reward for this state."
        return self.reward[state]

    def T(self, state, action):
        """Transition model.  From a state and an action, return a row in the matrix for next-state probability."""
        i = self.states.index(state)
        return self.prob[action][i, :]

    def P(self, state, action, next_state):
        "Derived from the transition model. For a state, an action and the next_state, return the probability of this transition."
        i = self.states.index(state)
        j = self.states.index(next_state)
        return self.prob[action][i, j]



    def actlist(self, state):
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
        if action not in self.actlist(state):
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
    def reward_traj(self, traj, flag):
        reward = self.reward
        st = traj[0]
        act = traj[1]
        if len(traj) >= 4:
            r = reward[st][act] + self.gamma * self.reward_traj(traj[2:], flag)
        else:
            return reward[st][act]
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
    def one_step_transition(self, st, act, st_lists, pro_lists):

        st_list = st_lists[st][act]
        pro_list = pro_lists[st][act]
        if all(x == 0 for x in pro_list):
            return None
        next_st = np.random.choice(len(st_list), 1, p=pro_list)[0]
        return st_list[next_st]

    def generate_sample(self, pi, trans, num_pairs):
        # pi here should be pi[st] = [pro1, pro2, ...]
        st_lists, pro_lists = stotrans_list(trans)
        traj = []
        st_index = np.random.choice(len(self.states), 1, p=self.init)[0]
        st = self.states[st_index]
        act_index = np.random.choice(len(self.actlist), 1, p=pi[st])[0]
        act = self.actlist[act_index]
        traj.append(st)
        traj.append(act)
        next_st = self.one_step_transition(st, act, st_lists, pro_lists)
        for _ in range(0, num_pairs):
            st = next_st
            # st_index = self.states.index(st)
            act_index = np.random.choice(len(self.actlist), 1, p=pi[st])[0]
            act = self.actlist[act_index]
            traj.append(st)
            traj.append(act)
            next_st = self.one_step_transition(st, act, st_lists, pro_lists)
        traj.append(next_st)
        return traj

def stotrans_list(transition):
    transition_list = {}
    transition_pro = {}
    for st in transition:
        transition_list[st] = {}
        transition_pro[st] = {}
        for act in transition[st]:
            transition_list[st][act] = {}
            transition_pro[st][act] = {}
            st_list = []
            pro_list = []
            for st_, pro in transition[st][act].items():
                st_list.append(st_)
                pro_list.append(pro)
            transition_list[st][act] = st_list
            transition_pro[st][act] = pro_list
    return transition_list, transition_pro






