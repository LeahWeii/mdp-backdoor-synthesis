from MDP import *
from Sample import *


def reward2list(reward, states, actions):
    reward_s = []
    for st in states:
        for act in actions:
            reward_s.append(reward[st][act])
    return np.array(reward_s)

class GC:
    def __init__(self, mdp, tau, policy, epsilon, approximate_flag):
        self.mdp = mdp
        self.st_len = len(mdp.states)
        self.act_len = len(mdp.actlist)
        self.x_size = self.st_len * self.act_len
        self.base_reward = reward2list(mdp.reward, mdp.states, mdp.actlist)
        self.x = np.zeros(self.x_size)
        self.tau = tau # temperature
        self.policy_m = self.convert_policy(policy)          #self.policy_m is a vector.
        self.epsilon = epsilon # approximation
        self.sample = Sample.SampleTraj(self.mdp)


    def convert_policy(self, policy):
        policy_m = np.zeros(self.x_size)
        i = 0
        for st in self.mdp.states:
            for j in range(self.mdp.actlist):
                policy_m[i] = policy[st][j]
                i += 1
        return policy_m
        #0 is not using approximate_policy, 1 is using approximate_policy
#Use sample to estimate the policy gradient
    def dJ_dtheta(self, Sample):
        # grdient of value function respect to theta
        # sample based method
        # returns dJ_dtheta_i, 1*NM matrix
        N = len(Sample.trajlist)
        grad = 0
        for rho in Sample.trajlist:
            # print("trajectory is:", rho)
            grad += self.drho_dtheta(rho) * self.mdp.reward_traj(rho, 0)
            # print(self.drho_dtheta(rho))
        print("grad is:", grad)
        return 1 / N * grad


    def drho_dtheta(self, rho):
        if len(rho) == 1:
            return np.zeros(self.x_size)
        st = rho[0]
        act = rho[1]
        rho = rho[2:]
        return self.dPi_dtheta(st, act) + self.drho_dtheta(rho)


    def dPi_dtheta(self, pol, st, act, tau):
        # dlog(pi)_dtheta
        grad = np.zeros(self.x_size)
        st_index = self.mdp.states.index(st)
        act_index = self.mdp.actlist.index(act)
        Pi = pol[st] # a vector of probability for taking different actions.
        # print("Pi:", Pi)
        for i in range(self.mdp.act_len):
            if i == act_index:
                grad[st_index * self.act_len + i] = 1 / self.tau * (1.0 - Pi[i])
            else:
                grad[st_index * self.act_len + i] = 1 / self.tau * (0.0 - Pi[i])
        return grad

def save_data(data):
    filename = "Jlist_modelfree200.pkl"
    file = open(filename, 'wb')
    pickle.dump(data, file)
    file.close()


def softmax(x, temperature =1.0):
    exp_x = np.exp((x - np.max(x)))/temperature  # Subtract max(x) for numerical stability
    return exp_x / np.sum(exp_x)


def valueIter(mdp, epsilon=0.1):
    # the transition matrix is sparse.
    pol = Policy(mdp.states, mdp.actlist)
    V = np.array([0.0 for s in mdp.states])
    while True:
        V_old = copy.deepcopy(V)
        for s in mdp.states:
            s_idx  = mdp.states.index(s)
            value  = np.array([mdp.reward[s][a]+mdp.gamma*mdp.prob[a][s_idx,:].dot(V_old) for a in mdp.actlist])
            pvec = softmax(value, temperature =1.0)
            pol.policy[s] = pvec
            V[s]  = np.inner(value, pvec) # the inner product given the updated value
        if np.linalg.norm(V-V_old, np.inf) <= epsilon:
            break
    return V, pol



def policyEval(mdp, pol, epsilon=0.1):
    # the transition matrix is sparse.
    V = np.array([0.0 for s in mdp.states])
    while True:
        V_old = copy.deepcopy(V)
        for s in mdp.states:
            s_idx  = mdp.states.index(s)
            value  = [mdp.reward[s][a]+ mdp.gamma*mdp.prob[a][s_idx,:].dot(V_old) for a in mdp.actlist]
            V[s_idx]  = np.inner(value, pol.policy[s])
        if np.linalg.norm(V-V_old, np.inf)<= epsilon:
            break
    return V


