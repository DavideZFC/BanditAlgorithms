from classes.environments.GaussianBandit import GaussianBandit
from classes.algos.ETC import ETC
from classes.algos.epsilon_greedy import epsilon_greedy
from classes.algos.UCB.UCB1 import UCB1
from classes.algos.UCB.UCB2 import UCB2
from classes.algos.UCB.MOSS import MOSS
from classes.algos.RLPE import RLPE
from functions.make_experiment import make_experiment
import numpy as np

means = np.array([1., 0.5, 0.])
env = GaussianBandit(means)

T = 1000

policies = []
labels = []

policy1 = UCB1(env.K(),T)
label1 = 'UCB1'

policy2 = epsilon_greedy(env.K(),T)
policy2.compute_best_param(delta=1.0)
label2 = 'epsilon-greedy-tuned'

policy3 = UCB2(env.K(), T)
label3 = 'UCB2'

policy4 = ETC(env.K(), T)
policy4.compute_best_param(delta=1.0)
label4 = 'ETC-tuned'

policy5 = MOSS(env.K(),T)
label5 = 'MOSS'

policy6 = RLPE(env.K(),T)
label6 = 'RLPE'

policies = [policy1, policy2, policy3, policy4, policy5, policy6]
labels = [label1, label2, label3, label4, label5, label6]

make_experiment(policies, env, T, seeds=20, labels=labels, exp_name='RLPE vs everyone short horizon')

