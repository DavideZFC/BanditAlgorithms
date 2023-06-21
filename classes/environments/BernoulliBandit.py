import numpy as np
from classes.environments.Bandit import Bandit

class BernoulliBandit(Bandit):

    def __init__(self, means):

        super().__init__(means)

        for i in range(len(means)):
            if (means[i] > 1 or means[i] < 0):
                raise ValueError('Means must be in [0,1]')

    def pull(self,a):
        ''' Pulls chosen arm, returning a sample from a Bernoulli distribution '''

        self.times_pulled[a] += 1
        return np.random.binomial(1, self.means[a])
        