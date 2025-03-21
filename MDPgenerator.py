
import random
import numpy as np
import string
from MDP import *
import pickle

# Function to generate a random probability distribution for a given list
def random_distribution(elements):
    # Generate random values for each element
    random_values = np.random.rand(len(elements))
    # Normalize the values to sum to 1
    probabilities = random_values / np.sum(random_values)
    rounded_probabilities = np.round(probabilities, 2)
    rounded_probabilities[-1] = 1 - np.sum(rounded_probabilities[:-1])
    return dict(zip(elements, rounded_probabilities))

def generateMDP(num_states, num_actions):
    """

    :param num_states: the number of states
    :param num_actions: the number of actions.
    :return: an MDP with randomly generated transition dynamics.
    """
    states = list(range(num_states)) #note that it uses ordered number to denote state
    alphabet = list(string.ascii_lowercase)
    actlist = alphabet[:num_actions]
    mdp = MDP()
    mdp.states = states
    mdp.actlist = actlist
    mdp.init = np.array(list(random_distribution(states).values()))
    N = len(mdp.states)
    for a in mdp.actlist:
        mdp.prob[a] = np.zeros((N, N))
    max_num = N # the possible outcomes can
    for s in states:
        for a in actlist:
            num_next_state = random.choice(range(1,max_num))
            next_states = random.sample(states, num_next_state)
            temp = random_distribution(next_states)
            for ns in next_states:
                mdp.prob[a][mdp.states.index(s), mdp.states.index(ns)] = temp[ns]
    #covert prob to trans
    for s in mdp.states:
        mdp.trans[s] = {}
        for a in mdp.actlist:
            mdp.trans[s][a] = {}
            for ns in mdp.states:
                # Get probability using matrix indices
                mdp.trans[s][a][ns] = mdp.prob[a][mdp.states.index(s), mdp.states.index(ns)]
    mdp.reward = {
        s: {a: random.uniform(0, num_states*num_actions) for a in actlist} # only allow positive rewards
        for s in states
    }
    return mdp

def get_rectangular_set_act(mdp, action, epsilon):
    """
    Generates an ε-rectangular set of transition functions.

    Parameters:
    P : numpy array (S, A, S)
        Original transition probability matrix where P[s, a, s'] = P(s' | s, a)
    epsilon : float
        Maximum allowable deviation from original probabilities

    Returns:
    P_min : numpy array (S, A, S)
        Lower bound of transition probabilities
    P_max : numpy array (S, A, S)
        Upper bound of transition probabilities
    """
    P = mdp[action] # the transition matrix given action a

    # Compute lower and upper bounds
    P_min = np.maximum(P - epsilon, 0)  # Ensure non-negative probabilities
    P_max = np.minimum(P + epsilon, 1)  # Ensure probabilities don't exceed 1

    # Normalize each transition probability set to sum to 1
    for s in mdp.states:
        P_min[s, :] /= P_min[s, :].sum() if P_min[s, :].sum() > 0 else 1
        P_max[s, :] /= P_max[s, :].sum() if P_max[s, :].sum() > 0 else 1
    return P_min, P_max

def get_rectangular_set(mdp, noise_level):
    """
      Compute the rectangular uncertainty set

    Parameters:
    - mdp: the original MDP.
    - noise_level: float, fraction of probability mass to redistribute (e.g., 0.1 for 10%)

    Returns:
    - P_min, P_max are both dictionary objects.
    P_min[a] is the lower bound of the Transition Probabilities for action a.
    P_max[a] is the upper bound on Transition Probabilities of action a.
    """
    P_min = {}
    P_max = {}
    for act in mdp.actlist:
        P_min[act], P_max[act] = get_rectangular_set_act(mdp, act, noise_level)
    return P_min, P_max

def sample_k_trans(mdp, K, epsilon):
    """

    :param mdp: original MDP
    :param K: K different transitions in rectangular uncertainty set
    :param noise_level:  float, fraction of probability mass to redistribute (e.g., 0.1 for 10%)
    :return:
    """
    samples = []
    for i in range(K):
        P_sample = {a:np.zeros_like(mdp.prob[a]) for a in mdp.actlist}
        for a in mdp.actlist:
            temp = np.copy(mdp.prob[a])
            # Generate perturbations within the ε-range
            perturbation = np.random.uniform(-epsilon, epsilon, size=temp.shape)
            temp  += perturbation  # Apply perturbation
        # Ensure probabilities remain in valid range [0,1]
            P_sample[a] = np.clip(temp, 0, 1)

        # Normalize each transition probability to sum to 1
            for s in mdp.states:
                P_sample[a][s, :] /= P_sample[a][s, :].sum() if P_sample[a][s, :].sum() > 0 else 1
        samples.append(P_sample)
    return samples




def read_from_file_MDP(fname):
    """
    This function takes the input file and construct an MDP based on the transition relations.
    The first line of the file is the list of states.
    The second line of the file is the list of actions.
    Starting from the second line, we have
    state, action, next_state, probability
    """
    f = open(fname, 'r')
    array = []
    for line in f:
        array.append(line.strip('\n'))
    f.close()
    mdp = MDP()
    state_str = array[0].split(",")
    mdp.states = [i for i in state_str]
    act_str = array[1].split(",")
    mdp.actlist = act_str
    mdp.prob = dict([])
    N = len(mdp.states)
    for a in mdp.actlist:
        mdp.prob[a] = np.zeros((N, N))
    for line in array[2: len(array)]:
        trans_str = line.split(",")
        state = trans_str[0]
        act = trans_str[1]
        next_state =  trans_str[2]
        p = float(trans_str[3])
        mdp.prob[act][mdp.states.index(state), mdp.states.index(next_state)] = p
    return mdp




if __name__ == "__main__":
    random.seed(0)
    mdp1= generateMDP(3,3)
    mdp1pickle= pickle.dumps(mdp1)
    testmdp1 = pickle.loads(mdp1pickle)

    print("State Space:")
    for state in testmdp1.states:
        print(state)

    # Print action space
    print("\nAction Space:")
    for action in testmdp1.actlist:
        print(action)

    # Print transitions
    print("\nProb:")
    for action in testmdp1.actlist:
        print(f"Action '{action}':")
        for i, state_from in enumerate(testmdp1.states):
            for j, state_to in enumerate(testmdp1.states):
                probability = testmdp1.prob[action][i, j]
                print(f"  From state '{state_from}' to state '{state_to}' with probability {probability:.2f}")

    print("\nTrans:")
    for action in testmdp1.actlist:
        print(f"Action '{action}':")
        for state_from in testmdp1.states:
            for state_to in testmdp1.states:
                trans = testmdp1.trans[state_from][action][state_to]
                print(f"  From state '{state_from}' to state '{state_to}' with probability {trans:.2f}")


    print("\nRewards:")
    for i, state in enumerate(testmdp1.states):
        for a in (testmdp1.actlist):
            reward = testmdp1.reward[i][a]
            print(f"  reward at state '{state}' and action '{action}' is {reward:.2f}")


    print("completing the MDP generation.")

    #test for function in MDP
    pi = {}
    for state in testmdp1.states:
        # Generate random probabilities for each action
        probs = [random.random() for _ in testmdp1.actlist]

        # Normalize probabilities to sum to 1
        total = sum(probs)
        normalized_probs = [p / total for p in probs]

        # Assign normalized probabilities to the state
        pi[state] = normalized_probs
    traj = testmdp1.generate_sample(pi,testmdp1.trans,3)
    print(traj)



