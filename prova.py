from classes.environments.GaussianBandit import GaussianBandit
from classes.algos.ETC import ETC
from classes.algos.epsilon_greedy import epsilon_greedy
from functions.make_experiment import make_experiment
import numpy as np

means = np.array([1.,0.])
env = GaussianBandit(means)

T = 100

policies = []
labels = []

policy1 = epsilon_greedy(env.K(),T)
label1 = 'epsilon-greedy'

policy2 = epsilon_greedy(env.K(),T)
policy2.compute_best_param(delta=1.0)
label2 = 'epsilon-greedy-tuned'

policy3 = ETC(env.K(), T)
label3 = 'ETC'

policy4 = ETC(env.K(), T)
policy4.compute_best_param(delta=1.0)
label4 = 'ETC-tuned'

policies = [policy1, policy2, policy3, policy4]
labels = [label1, label2, label3, label4]

make_experiment(policies, env, T, seeds=5, labels=labels, exp_name='ETC_vs_epsG')

