
import random
import numpy as np
import string
from MDP import *
import pickle
import numpy as np

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
    states = range(num_states) #note that it uses ordered number to denote state
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
        s: {a: random.uniform(-num_states*num_actions, num_states*num_actions) for a in actlist}
        for s in states
    }
    return mdp


def add_noise_to_transition(transition_function, noise_level):
    """
    Adds noise to an MDP transition function.

    Parameters:
    - transition_function: dict[state][action] -> numpy array of probabilities
    - states: list of all possible states
    - noise_level: float, fraction of probability mass to redistribute (e.g., 0.1 for 10%)

    Returns:
    - noisy_transition_function: dict with noise added
    """
    noisy_transition_function = {}
    num_states = len(trans)

    for state in transition_function:
        noisy_transition_function[state] = {}
        for action in transition_function[state]:
            # Get original probabilities
            original_probs = transition_function[state][action]

            # Redistribute noise uniformly
            uniform_noise = np.full(num_states, noise_level / num_states)
            adjusted_probs = (1 - noise_level) * original_probs + uniform_noise

            # Normalize to ensure sum is 1
            adjusted_probs /= np.sum(adjusted_probs)

            # Update noisy transition function
            noisy_transition_function[state][action] = adjusted_probs

    return noisy_transition_function
def generate_k_trans(k, trans):
    trans_list = []
    noise_level_list = random.sample(range(0.0001, k/10), k)
    for noise_level in noise_level_list:
        trans_list.append(add_noise_to_transition(trans, noise_level))





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



