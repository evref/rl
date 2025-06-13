import gymnasium as gym

from agents import QLearningAgent, FrozenLakeAgent


MAX_STEPS = 100

def run_sim(env: gym.Env, agent: FrozenLakeAgent):
    obs, _ = env.reset()

    episode_over = False
    while not episode_over:
        action = agent.get_action(env, obs)

        last_obs = obs

        obs, reward, terminated, truncated, _ = env.step(action)
        agent.update(action, reward, obs, last_obs)

        episode_over = terminated or truncated

    env.close()
        
    if reward > 0:
        print("WE DIDIT BOYS")


agent = QLearningAgent(observation_space_size=16, 
                       action_space_size=4, 
                       learning_rate=0.5)

env = gym.make('FrozenLake-v1', 
               desc=None, 
               map_name="4x4", 
               is_slippery=False, 
               max_episode_steps=MAX_STEPS)
for i in range(150):
    print(f"Running simulation # {i}")
    run_sim(env, agent)


env = gym.make('FrozenLake-v1', 
               desc=None, 
               map_name="4x4", 
               is_slippery=False, 
               render_mode="human", 
               max_episode_steps=MAX_STEPS)

run_sim(env, agent)