from classes.environments.GaussianBandit import GaussianBandit
import numpy as np

means = np.array([1.10,0.12])
env = GaussianBandit(means)
print(env.pull(1))
