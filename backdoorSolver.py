from MDPgenerator import *
import math

def get_augMDP(mdp, trigger, sampledTrans):
    """

    :param mdp: the original MDP.
    :param trigger: the finite-state trigger
    :param sampledTrans:  the sampled set of transition functions.
    :return:
    """

def update_trans_attacker(mdp, mdp_attacker, trans_samples, pi0):
    #pi0[s] = [p1,p2,p3,...]
    states = mdp_attacker.states
    actlist = mdp.actlist

    for s in states:
        mdp_attacker.trans[s] = {}
        for a1 in mdp_attacker.actlist:
            mdp_attacker.trans[s][a1] = {}
            for ns in states:
                mdp_attacker.trans[s][a1][ns] = 0
                for a0 in actlist:
                    # Check if trans_samples is a valid probability distribution
                    if not math.isclose(sum(trans_samples[a1][s][a0].values()), 1.0, rel_tol=1e-6):
                        print(f"Warning: trans_samples[{a1}][{s}][{a0}] does not sum to 1")
                    mdp_attacker.trans[s][a1][ns] += pi0[s][actlist.index(a0)] * trans_samples[a1][s][a0][ns]
    # Test if probabilities sum to 1
    # Test if probabilities sum to 1
    for s in states:
        for a1 in mdp_attacker.actlist:
            prob_sum = sum(mdp_attacker.trans[s][a1].values())
            if not math.isclose(prob_sum, 1.0, rel_tol=1e-6):
                print(f"Warning: Probabilities for state {s}, action {a1} sum to {prob_sum}, not 1.0")
                print(f"Probabilities: {mdp_attacker.trans[s][a1]}")
                print(f"pi0[{s}] = {pi0[s]}")
                print(f"Sum of pi0[{s}] = {sum(pi0[s])}")
            else:
                print(f"Test passed for state {s}, action {a1}")


if __name__ == "__main__":
    trans, mdp, mdp_attacker = get_aug_mdp(3,3,4, 0.2, 0)


    pi = {}
    for state in mdp_attacker.states:
        # Generate random probabilities for each action
        probs = [random.random() for _ in mdp_attacker.actlist]

        # Normalize probabilities to sum to 1
        total = sum(probs)
        normalized_probs = [p / total for p in probs]

        # Assign normalized probabilities to the state
        pi[state] = normalized_probs
        
    update_trans_attacker(mdp, mdp_attacker, trans, pi)

    print("\nTrans:")
    for action in mdp_attacker.actlist:
        print(f"Action '{action}':")
        for state_from in mdp_attacker.states:
            for state_to in mdp_attacker.states:
                trans = mdp_attacker.trans[state_from][action][state_to]
                print(f"  From state '{state_from}' to state '{state_to}' with probability {trans:.2f}")
    