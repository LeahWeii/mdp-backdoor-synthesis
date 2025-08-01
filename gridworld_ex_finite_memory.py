from  GridWorld import GridWorld
from MDP import *
import pickle
from FSCTrigger_gw import FSCTriggerGW, FSCTriggerGW_finite
import backdoorSolver_PO
import os

def createGridWorldBarrier_new2(stoPar):
    gamma = 0.99
    tau = 0.1
    goallist = [(4, 3), (0, 5)]
    barrierlist = []
    initial_state = (0, 0)
    fakelist = [(5, 4), (4, 1)]  # case 1
    # fakelist = [(0, 2), (5, 3)] #case 2
    IDSlist = [(0, 4), ( 2,1 ), (3,2), (3, 3), (5, 4)]
    gridworld = GridWorld(6, 6, stoPar, initial_state, fakelist, goallist, IDSlist, barrierlist, gamma, tau)
    reward = []
    # V, policy = gridworld.get_policy_entropy(reward, 1)
    # V_def = gridworld.policy_evaluation(reward, 0, policy)
    # return gridworld, V_def, policy
    return gridworld


def createGridWorldBarrier_adv(stoPar):
    gamma = 0.95
    tau = 0.1
    # goallist = [(2, 5), (5, 2)]
    fakelist = [(4, 3), (0, 5)]
    barrierlist = []
    initial_state = (0, 0)
    goallist = [(5, 4), (4, 1)]  # case 1 # the attacker want to reach (5, 4) and (4, 1)
    # fakelist = [(0, 2), (5, 3)] #case 2
    # fakelist = [(0, 2), (5, 3)] #case 2
    IDSlist = [(0, 4), (1, 2), (2, 3), (3, 3), (5, 4)]
    gridworld = GridWorld(6, 6, stoPar, initial_state, fakelist, goallist, IDSlist, barrierlist, gamma, tau)
    reward = []
    # V, policy = gridworld.get_policy_entropy(reward, 1)
    # V_def = gridworld.policy_evaluation(reward, 0, policy)
    # return gridworld, V_def, policy
    return gridworld

def get_zerosum_reward(gridworld):
    reward = gridworld.leader_reward()
    adv_reward = {st: {act: 0 for act in gridworld.actions} for st in
                  gridworld.states}
    for st in reward.keys():
        for act in reward[st].keys():
            adv_reward[st][act] = - reward[st][act] #Todo
    return adv_reward

def get_adv_cost(trigger):
    adv_cost = {st: {act: 0 for act in trigger.actlist} for st in
                trigger.states}
    for st in trigger.states:
        for act in trigger.actlist:
            if act == 0: # the default dynamic
                adv_cost[st][act] = 0
            else:
                adv_cost[st][act] = 1
    return adv_cost

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

def plot_final_vs_epsilon(epsilons, V0_final, V1_final, V0_final_attack, path="./results"):
    import os
    import matplotlib.pyplot as plt
    os.makedirs(path, exist_ok=True)
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    # First subplot: V0_final and V0_final_attack
    axs[0].plot(epsilons, V0_final, marker='o', label='V0_final')
    axs[0].plot(epsilons, V0_final_attack, marker='o', color='green', label='V0_final_attack')
    axs[0].set_xlabel('Epsilon')
    axs[0].set_ylabel('V0')
    axs[0].set_title('V0 and V0 under attack vs Epsilon')
    axs[0].legend()
    axs[0].grid(True)
    # Second subplot: V1_final
    axs[1].plot(epsilons, V1_final, marker='o', color='orange', label='V1_final')
    axs[1].set_xlabel('Epsilon')
    axs[1].set_ylabel('V1')
    axs[1].set_title('V1 vs Epsilon')
    axs[1].legend()
    axs[1].grid(True)
    plt.tight_layout()
    fig.savefig(os.path.join(path, "final_vs_epsilon.png"))
    plt.show()
    print(f"Last V0_final_attack: {V0_final_attack[-1]}")

# from backdoorSolver_Adam import plot_results
def plot_from_pickled_results(path, episodes=None, lb=None):
    """
    Read V0_iter, V1_iter, V0_attack from pickled files in path and plot them in three subplots.
    If episodes or lb are not provided, they are inferred from the data or skipped.
    """
    with open(os.path.join(path, "V0.pkl"), "rb") as f0:
        V0_iter = pickle.load(f0)
    with open(os.path.join(path, "V1.pkl"), "rb") as f1:
        V1_iter = pickle.load(f1)
    with open(os.path.join(path, "V0_attack.pkl"), "rb") as f2:
        V0_iter_attack = pickle.load(f2)
    if episodes is None:
        episodes = min(len(V0_iter), len(V1_iter), len(V0_iter_attack))
    # Plot

    plot_results(path, V0_iter, V1_iter, V0_iter_attack, episodes, lb)

def plot_final_from_pickled_results(basepath):
    epsilons = [0.1, 0.15, 0.2, 0.25, 0.3]
    V0_final = []
    V1_final = []
    V0_final_attack = []
    for eps in epsilons:
        path = os.path.join(basepath, f"epsilon_{eps}")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "V0.pkl"), "rb") as f0:
            V0_iter = pickle.load(f0)
        with open(os.path.join(path, "V1.pkl"), "rb") as f1:
            V1_iter = pickle.load(f1)
        with open(os.path.join(path, "V0_attack.pkl"), "rb") as f2:
            V0_iter_attack = pickle.load(f2)
        V0_final.append(V0_iter[-1] if len(V0_iter) > 0 else None)
        V1_final.append(V1_iter[-1] if len(V1_iter) > 0 else None)
        V0_final_attack.append(V0_iter_attack[-1] if len(V0_iter_attack) > 0 else None)
    plot_final_vs_epsilon(epsilons, V0_final, V1_final, V0_final_attack)


def batch_test_switchingGradient(mdp, adv_reward, trigger, augmdp, K, base_path, episodes=1000, lr=0.01, tolerance=1e-2):
    """
    Run switchingGradient_no_marginalization for a batch of epsilon values and save results in separate folders.
    After running, compare the final V0_iter[-1] and V1_iter[-1] for each epsilon and return two lists.
    """

    epsilons = [0.1, 0.15, 0.2, 0.25, 0.3]
    V0_final = []
    V1_final = []
    V0_final_attack = []
    for eps in epsilons:
        path = os.path.join(base_path, f"epsilon_{eps}")
        os.makedirs(path, exist_ok=True)
        print(f"Running for epsilon={eps}, results will be saved in {path}")
        backdoorSolver_PO.switchingGradient_no_marginalization(
            mdp, eps, adv_reward, trigger, augmdp, K, path,
            episodes=episodes, lr=lr, tolerance=tolerance
        )
        # Load results
        try:
            with open(os.path.join(path, "V0.pkl"), "rb") as f0:
                V0_iter = pickle.load(f0)
            with open(os.path.join(path, "V1.pkl"), "rb") as f1:
                V1_iter = pickle.load(f1)
            with open(os.path.join(path, "V0_attack.pkl"), "rb") as f2:
                V0_iter_attack = pickle.load(f2)
            V0_final.append(V0_iter[-1] if len(V0_iter) > 0 else None)
            V1_final.append(V1_iter[-1] if len(V1_iter) > 0 else None)
            V0_final_attack.append(V0_iter_attack[-1] if len(V0_iter_attack) > 0 else None)
        except Exception as e:
            print(f"Error loading results for epsilon={eps}: {e}")
            V0_final.append(None)
            V1_final.append(None)
            V0_final_attack.append(None)
    plot_final_vs_epsilon(epsilons, V0_final, V1_final, V0_final_attack)
    return V0_final, V1_final, V0_final_attack

if __name__ == "__main__":
    #plot_from_pickled_results('./gridworld_ex/epsilon_0.1', episodes=10000, lb=10.15)
    # plot_final_from_pickled_results('./gridworld_ex/')
    # gridworld, V_def, policy = createGridWorldBarrier_new2()
    stoPar = 0.1
    gridworld = createGridWorldBarrier_new2(stoPar)
    st = gridworld.states

    stoPar_perturbed= [0.1, 0.2]
    gridworlds_perturbed =  [ createGridWorldBarrier_new2(sto) for sto in stoPar_perturbed]
    adversary_reward = get_zerosum_reward(gridworld)
    #
    # with open("gridworld_ex/gridworld.pkl", "wb") as file1:  # "wb" means write in binary mode
    #     pickle.dump(gridworld.trans, file1)
    #
    # with open("gridworld_ex/gridworlds_perturbed.pkl", "wb") as file2:  # "wb" means write in binary mode
    #     pickle.dump(gridworlds_perturbed, file2)
    memory_length = 1
    k = len(gridworlds_perturbed)
    trigger = FSCTriggerGW_finite(gridworld, k, memory_length)
    # constructing the transition function of the trigger.
    adversary_cost = get_adv_cost(trigger)

    p_obs = 0.8
    augmdp = backdoorSolver_PO.get_augMDP(gridworld, trigger,  gridworlds_perturbed, adversary_reward, adversary_cost)
    # This augmented MDP has very sparse transition matrix. should use sparse matrix for future.
    # backdoorSolver_Adam.switchingGradient(mdp1, adv_reward, trigger, augmdp, k)

    # warm-starting part
    epsilon = 0.1
    episodes_num =1


    path = './gridworld_ex_fm_po/p' +str(p_obs) + '_epsilon'+str(epsilon)
    print(path)
    path = './gridworld_ex_fm_po/test'
    os.makedirs(path,exist_ok=True)
    # batch_test_switchingGradient(gridworld, adversary_reward, trigger, augmdp, k, './gridworld_ex', episodes=episodes_num, lr=0.01, tolerance=1e-2)
    backdoorSolver_PO.switchingGradient_no_marginalization(gridworld,  epsilon, adversary_reward, trigger, augmdp, k, path, episodes_num)

    path_a = path + '/ablation_pi1'
    os.makedirs(path_a,exist_ok=True)
    # batch_test_switchingGradient(gridworld, adversary_reward, trigger, augmdp, k, './gridworld_ex', episodes=episodes_num, lr=0.01, tolerance=1e-2)
    backdoorSolver_PO.ablation_pi1(gridworld,  epsilon, adversary_reward, trigger, augmdp, k, path_a, episodes_num)

    print("complete ...")