from MDP import *



def reward2list(reward, states, actions):
    reward_s = []
    for st in states:
        for act in actions:
            reward_s.append(reward[st][act])
    return np.array(reward_s)



def softmax(x, temperature =1.0):
    exp_x = np.exp((x - np.max(x)))/temperature  # Subtract max(x) for numerical stability
    return exp_x / np.sum(exp_x)

def valueIterHardmax(mdp, epsilon=0.01):
    # the transition matrix is sparse.
    pol = Policy(mdp.states, mdp.actlist)
    V = np.array([0.0 for s in mdp.states])
    while True:
        V_old = copy.deepcopy(V)
        for s in mdp.states:
            s_idx  = mdp.states.index(s)
            values = np.array([
                mdp.reward[s][a] + mdp.gamma * mdp.prob[a][s_idx, :].dot(V_old)
                for a in mdp.actlist
            ])
            best_action_idx = np.argmax(values)
            pvec = np.zeros(len(mdp.actlist))
            pvec[best_action_idx] = 1.0  # Hardmax: 1 for best action, 0 for others
            pol.policy[s] = pvec
            V[s_idx] = values[best_action_idx]  # Value under greedy policy
        if np.linalg.norm(V - V_old, np.inf) <= epsilon:
            break
    return V, pol

def valueIter(mdp, temperature =1,  epsilon=0.01):
    # the transition matrix is sparse.
    pol = Policy(mdp.states, mdp.actlist)
    V = np.array([0.0 for s in mdp.states])
    while True:
        V_old = copy.deepcopy(V)
        for s in mdp.states:
            s_idx  = mdp.states.index(s)
            value  = np.array([mdp.reward[s][a]+mdp.gamma*mdp.prob[a][s_idx,:].dot(V_old) for a in mdp.actlist])
            pvec = softmax(value, temperature)
            pol.policy[s] = pvec
            V[s_idx]  = np.inner(value, pvec) # the inner product given the updated value
        if np.linalg.norm(V-V_old, np.inf) <= epsilon:
            break
    return V, pol




def policyEval(mdp, pol,  epsilon=0.01, reward=None):
    # the transition matrix is sparse.
    if reward == None:
        reward = mdp.reward
    V = np.array([0.0 for s in mdp.states])
    while True:
        V_old = copy.deepcopy(V)
        for s in mdp.states:
            s_idx  = mdp.states.index(s)
            value  = [reward[s][a]+ mdp.gamma*mdp.prob[a][s_idx,:].dot(V_old) for a in mdp.actlist]
            V[s_idx]  = np.inner(value, pol.policy[s])
        if np.linalg.norm(V-V_old, np.inf)<= epsilon:
            break
    return V


