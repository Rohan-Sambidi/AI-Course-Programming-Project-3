The complete project description and instructions are available in the [index.html](doc/index.html) file.

# Tasks
1a. Implement a function to compute Q-values (get_q_value) using Q-learning update rules in qlearning.py.  
1b. Implement a function for iteratively computing cumulative reward (compute_cumulative_reward) in qlarning.py.  
1c. Implement a function for computing a decayed value for epsilon (get_epsilon) in qlearning.py. The decay routine should reduce the current epsilon value by 1%. The minimum value that the function returns must be 0.01.  

2a. Complete the implementation of learn() function in qlearning.py. In this function you are supposed to compute a policy to organize books using Q-learning. Use an epsilon-greedy policy, which selects at each state, the best action according to current Q values with probability 1 - epsilon and selects a random action with probability epsilon. Start with Q values as zero for all state-action pairs.  
2b. Run the program with the q-value and decayed epsilon value computed and plot the resulting episode vs. reward plots.
