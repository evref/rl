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
        max_reward_action = np.argmax(self.q_table[obs])
        if (self.q_table[obs, max_reward_action] == 0):
            #print("No value in qtable, I pick random!!")
            return env.action_space.sample()
        else:
            #print("I know exactly what to do!")
            return max_reward_action
        
    def update(self, action: int, reward: float, obs: int, last_obs: int):
        self.q_table[last_obs, action] += self.lr * (reward + self.discount_factor * max(self.q_table[obs]) - self.q_table[last_obs, action])