import numpy as np
from classes.algos.UCB.ucb import ucb

class MOSS(ucb):

    def choose_arm(self):
        '''
            Returns the choosen arm using the upper confidence bound approach
        '''
        if min(self.times_pulled)  == 0:
            return np.argmin(self.times_pulled)
        
        # computes mean        
        expected_return = self.rewards/self.times_pulled

        # computes upper bound
        aux = np.clip(self.T/(self.n_arms*self.times_pulled), a_min=1, a_max=10000)
        upper_bound = np.sqrt(4/self.times_pulled*np.log(aux))
        
        return np.argmax(expected_return+upper_bound)