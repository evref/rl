import gymnasium as gym
import numpy as np

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="human", max_episode_steps=100)
observation, info = env.reset()

num_states = env.observation_space.n
num_actions = env.action_space.n
qtable = np.zeros((num_states, num_actions))


episode_over = False
while not episode_over:
    max_reward_action = np.argmax(qtable[observation])
    if (qtable[observation, max_reward_action] == 0):
        print("No value in qtable, I pick random!!")
        action = env.action_space.sample()
    else:
        print("I know exactly what to do!")
        action = max_reward_action

    observation, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated

env.close()
if reward > 0:
    print("HE DIDIT BOYS")