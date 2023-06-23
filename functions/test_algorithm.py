import numpy as np

def test_algorithm(policy, env, T, seeds, first_seed=1):
    ''' Function to test a policy on a given environment 
        policy = policy to be tested
        env = environment
        T = number of time steps to make the experiment

        Returns:
        regret_matrix = a matrix of dimension seeds x T having as rows the sequence of cumulative regret for any given seed.
    '''

    regret_matrix = np.zeros((seeds, T))
    np.random.seed(first_seed)

    for seed in range(seeds):

        policy.reset()
        env.reset()
        for t in range(1,T):
            arm = policy.choose_arm()
            reward = env.pull(arm)

            # update regret matrix
            regret_matrix[seed, t] = env.regret()

            # update policy
            policy.update(arm, reward)
    
    return regret_matrix

