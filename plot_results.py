path_list = ['ex_degrade_05_adv', 'ex_degrade_05_noco', 'ex_degrade_10_adv', 'ex_degrade_10_noco']
import pickle

with open(f"{path_list[1]}/V0.pkl", "rb") as file_v0:  # "rb" means read in binary mode
    V0_iter = pickle.load(file_v0)
with open(f"{path_list[1]}/V1.pkl", "rb") as file_v1:  # "rb" means read in binary mode
    V1_iter = pickle.load(file_v1)

with open(f"{path_list[1]}/v_original.pkl", "rb") as file_or:  # "rb" means read in binary mode
    V1_original = pickle.load(file_or)

import matplotlib.pyplot as plt

plt.plot(V0_iter, label=r'$V_0(\pi_0^t, M )$')
plt.plot(V1_iter, label=r'$V_1(\pi_0^t, \pi_1^t, \mathcal{M})$')
# Optional: Add labels, title, legend
lb = 91*0.95
plt.axhline(y=lb, color='r', linestyle='--', linewidth=2, label='lower bound:'+str(lb))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Iterations")
plt.legend()
plt.ylabel("Value for Original Reward/Attack Reward")
plt.title("")
plt.show()
plt.savefig(f"{path}/plot_v_iter.png")