from classes.environments.GaussianBandit import GaussianBandit
from classes.algos.ETC import ETC
from classes.algos.epsilon_greedy import epsilon_greedy
import numpy as np

means = np.array([1.,0.])
env = GaussianBandit(means)

T = 100
policy = epsilon_greedy(2,T)
policy.compute_best_param(delta=1)

for t in range(T):
    arm = policy.choose_arm()
    print(arm)
    reward = env.pull(arm)
    policy.update(arm, reward)

print(env.regret())
