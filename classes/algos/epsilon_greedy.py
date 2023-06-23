import numpy as np

class epsilon_greedy:
    def __init__(self, n_arms, T=1):
        ''' Inintializes the algorithm
            n_arms = number of arms in the bandit problem
            T = time horizon
        '''

        self.n_arms = n_arms
        self.rewards = np.zeros(n_arms)
        self.times_pulled = np.zeros(n_arms)
        self.T = T
        self.epsilon_routine = T**(-1/3)*np.ones(T)
        self.t = 0

    def compute_best_param(self, delta=0):
        ''' Tunes epsilon parameter of the algorithm
            delta = gap between best arm and second
        '''
        self.epsilon_routine = self.n_arms/((1+np.arange(self.T))*delta**2)

    def reset(self):
        ''' Reinitializes the variables
        '''

        self.rewards = np.zeros(self.n_arms)
        self.times_pulled = np.zeros(self.n_arms)
        self.t = 0


    def choose_arm(self):
        ''' Chooses which arm to pull
        '''
        p = min(self.epsilon_routine[self.t],1)
        if (np.random.binomial(1,p)) or min(self.times_pulled)  == 0:
            return np.argmin(self.times_pulled)
        else:
            return np.argmax(self.rewards/self.times_pulled)

    def update(self, arm, reward):
        ''' Updates internal variables to take into account the reward received at this round
            arm = arm pulled
            reward = reward received when pulling the corresponding arm 
        '''

        self.times_pulled[arm] += 1
        self.rewards[arm] += reward
        self.t += 1