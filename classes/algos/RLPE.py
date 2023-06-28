import numpy as np


class RLPE:

    def __init__(self, n_arms, T):
        ''' Inintializes the algorithm
            n_arms = number of arms in the bandit problem
            T = time horizon
        '''
        self.n_arms = n_arms
        self.times_pulled = np.zeros(self.n_arms)
        self.rewards = np.zeros(self.n_arms)

        # History of how many times an arm was ranked first
        self.hist_first = np.zeros(self.n_arms)
        self.T = T

        # define loggrid
        logt = int(np.log(T))
        self.loggrid = self.T**np.linspace(1/2,1,logt)
        print('Loggrid defined')
        print(self.loggrid)
        self.logindex = 0

        self.t = 0

        # the set of active arms represents the candidates for the best arm
        self.active_arms = range(n_arms)
    
    def reset(self):
        self.times_pulled = np.zeros(self.n_arms)
        self.rewards = np.zeros(self.n_arms)
        self.hist_first = np.zeros(self.n_arms)
        self.logindex = 0
        self.t = 0
        self.active_arms = range(self.n_arms)

    def get_least_pulled(self):
        ''' This function returns the least pulled arm between all arms that are active now'''


        pulls = self.T
        candidate_arm = 0
        for j in range(len(self.active_arms)):
            arm = self.active_arms[j]
            if (self.times_pulled[arm] < pulls):
                candidate_arm = arm
                pulls = self.times_pulled[arm]
        return candidate_arm, pulls
   
    def choose_arm(self):

        # pull the less pulled arm in the set of new arms
        candidate_arm, _ = self.get_least_pulled()

        return int(candidate_arm)
    
    def update(self, arm, reward):
        ''' Updates internal variables to take into account the reward received at this round
            arm = arm pulled
            reward = reward received when pulling the corresponding arm 
        '''
        self.times_pulled[arm] += 1
        self.rewards[arm] += reward
        self.t += 1

        if min(self.times_pulled) > 0:
            first = np.argmax(self.rewards/self.times_pulled)
            self.my_update(first)
        

    def my_update(self, first):

        _, pulls = self.get_least_pulled()

        # this is triggered only if all the arms have been pulled the same number of times
        if pulls == np.max(self.times_pulled):            
            self.hist_first[int(first)] += 1

        if pulls > self.loggrid[self.logindex]: 
            # critical exponent
            alpha = np.log(pulls)/np.log(self.T) - 1/2

            self.active_arms = np.array([j for j in range(self.n_arms) if self.hist_first[j] >= int(self.T**(2*alpha))])
            
            if self.logindex < len(self.loggrid)-1:
                self.logindex += 1
