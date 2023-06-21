import numpy as np
from classes.environments.Bandit import Bandit

class GaussianBandit(Bandit):

    def __init__(self, means, std = 1):
        ''' Receives the standard deviation, which we assume to be the same for every arm '''

        super().__init__(means)
        self.std = std

    def pull(self,a):
        ''' Pulls chosen arm, returning a sample from a Gaussian distribution '''

        self.times_pulled[a] += 1
        return np.random.normal(self.means[a], self.std)
        