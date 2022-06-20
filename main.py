import gym
 
from src.agent import Agent


env = gym.make("MountainCarContinuous-v0")

model = Agent(env)
model.learn()

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()
