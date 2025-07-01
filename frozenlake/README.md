# The FrozenLake Challenge from the Gymnasium, an API for RL
The FrozenLake challenge puts the agent into a 2D-grid of NxM size. The agent receives a reward of 1 by walking into the tile of the present and a reward of 0 otherwise. There also exists ice puddles which the agent can't escape if it falls into. 

Action space:
- 0: LEFT
- 1: DOWN
- 2: RIGHT
- 3: UP

Observation space:
- The 2D grid, each represented by an integer value.

More about it can be found on the website: https://gymnasium.farama.org/environments/toy_text/frozen_lake/

## Q-Learning Agent
This agent is taught to solve a specific FrozenLake Challenge grid using the Q-Learning method, defined by the following equation:

$Q_{new}(s_t,a_t)=Q(s_t,a_t)+\alpha*(r_t+\gamma*max_aQ(s_{t+1},a)-Q(s_t,a_t))$

Where 
- $s_t$ is the observation state at timestep $t$
- $a_t$ is the action state at timestep $t$
- $r_t$ is the reward at timestep $t$
- $\alpha$ is the learning rate
- $\gamma$ is the discount rate
- $Q(s,a)$ is the reward score of an observation state $s$ and an action state $a$ in the Q-Table 

Helpful Medium post about it: https://medium.com/data-science/q-learning-for-beginners-2837b777741
