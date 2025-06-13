import gymnasium as gym
import numpy as np
from abc import ABC, abstractmethod


class FrozenLakeAgent(ABC):
    @abstractmethod
    def get_action(self, obs: int) -> int:
        pass

    @abstractmethod
    def update(self):
        pass

class QLearningAgent(FrozenLakeAgent):
    """
    Initialize Q-Learning Agent with start values.
    
    Args:
        observation_space_size: Size of observation space
        action_space_size: Size of action space
        learning_rate: Learning rate
        discount_factor: Discount factor
    """
    def __init__(self,
                 observation_space_size: int,
                 action_space_size: int,
                 learning_rate: float,
                 discount_factor: float = 0.95,
    ):
        self.q_table = np.zeros((observation_space_size, action_space_size))

        self.lr = learning_rate
        self.discount_factor = discount_factor

    def get_action(self, env: gym.Env, obs: int) -> int:
        """
        Given the environment and an observation returns an action.
        Looks in the Q-Table and takes the action with the highest 
        reward score. If all actions have reward scores of 0 it 
        picks a random direction.
        
        Args:
            env: The gym environment
            obs: Current location of player

        Output:
            The action to take, int between 0-3
        """
        max_reward_action = np.argmax(self.q_table[obs])
        if (self.q_table[obs, max_reward_action] == 0):
            #print("No value in qtable, I pick random!!")
            return env.action_space.sample()
        else:
            #print("I know exactly what to do!")
            return max_reward_action
        
    def update(self, action: int, reward: float, obs: int, last_obs: int):
        """
        Updates the Q-Table of the agent using the standard Q-learning-
        algorithm. The learning rate determines the speed of learning while
        the discount factor considers determines how much to value rewards
        further away.
        
        Args:
            action: The action taken to get to obs
            reward: The reward received from going to obs
            obs: The observation state that is result of previous action
            last_obs: The previous observation state (t-1)
        """
        self.q_table[last_obs, action] += self.lr * (reward + self.discount_factor * max(self.q_table[obs]) - self.q_table[last_obs, action])