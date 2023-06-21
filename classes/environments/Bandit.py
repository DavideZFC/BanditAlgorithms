import numpy as np

class Bandit:

    def __init__(self, means):
        ''' General  class for bandit algorithms, takes as input a vector of the means of each arm, as a numpy array '''

        self.means = means
        self.times_pulled = np.zeros_like(self.means)

    def reset(self):
        ''' Initializes to zero the number of times each arm have been pulled '''

        self.times_pulled = np.zeros_like(self.means)

    def K(self):
        ''' Returns the number of arms '''

        return len(self.means)
    
    def regret(self):
        ''' Computes the expected regret incurred so far '''

        n = np.sum(self.times_pulled)
        best_arm = np.max(self.means)

        regret = n*best_arm
        for i in range(len(self.means)):
            regret -= self.times_pulled[i]*self.means[i]
        return regret