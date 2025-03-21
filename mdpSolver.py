from MDP import *
from Sample import *



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
    # print("grad is:", grad)
    return 1 / N * grad


def drho_dtheta(self, rho):
    if len(rho) == 1:
        return np.zeros(self.x_size)
    st = rho[0]
    act = rho[1]
    rho = rho[2:]
    return self.dPi_dtheta(st, act) + self.drho_dtheta(rho)


def dPi_dtheta(self, st, act):
    # dlog(pi)_dtheta
    grad = np.zeros(self.x_size)
    st_index = self.mdp.states.index(st)
    act_index = self.mdp.actions.index(act)
    Pi = self.policy[st]
    # print("Pi:", Pi)
    for i in range(self.act_len):
        if i == act_index:
            grad[st_index * self.act_len + i] = 1 / self.tau * (1.0 - Pi[i])
        else:
            grad[st_index * self.act_len + i] = 1 / self.tau * (0.0 - Pi[i])
    # grad is a vector x_size * 1

    return grad




