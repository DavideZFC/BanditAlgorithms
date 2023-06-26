import numpy as np
from classes.algos.UCB.ucb import ucb

class UCB1(ucb):

    def __init__(self, n_arms, T):
        super().__init__(n_arms, T)

    def choose_arm(self):
        '''
            Returns the choosen arm using the upper confidence bound approach
        '''
        if min(self.times_pulled)  == 0:
            return np.argmin(self.times_pulled)
        
        # computes mean
        expected_return = self.rewards/self.times_pulled

        # computes upper bound
        delta = 1/self.T**2
        upper_bound = np.sqrt(2*np.log(1/delta)/self.times_pulled)

        return np.argmax(expected_return+upper_bound)