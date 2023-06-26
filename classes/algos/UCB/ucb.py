import numpy as np

class ucb:
    '''
    This class is only a parent class for the algorithms defined in the same folder
    '''

    def __init__(self, n_arms, T=1):
        ''' Inintializes the algorithm
            n_arms = number of arms in the bandit problem
            T = time horizon
        '''

        self.n_arms = n_arms
        self.rewards = np.zeros(n_arms)
        self.times_pulled = np.zeros(n_arms)
        self.T = T
        self.t = 0

    def compute_best_param(self, delta=0):
        pass

    def reset(self):
        ''' Reinitializes the variables
        '''

        self.rewards = np.zeros(self.n_arms)
        self.times_pulled = np.zeros(self.n_arms)
        self.t = 0


    def update(self, arm, reward):
        ''' Updates internal variables to take into account the reward received at this round
            arm = arm pulled
            reward = reward received when pulling the corresponding arm 
        '''

        self.times_pulled[arm] += 1
        self.rewards[arm] += reward
        self.t += 1