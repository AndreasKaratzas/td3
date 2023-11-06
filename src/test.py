
import sys
sys.path.append('../')

from lib.agent import Agent


def test(agent: Agent, demo_episodes: int = None):
    for demo_ep in range(demo_episodes if demo_episodes is not None else agent.demo_episodes):
        state, done, episode_return, episode_length = agent.test_env.reset(seed=agent.seed), False, 0, 0
        while not(done or (episode_length == agent.max_ep_len)):
            # Take deterministic actions at test time
            state, reward, done, _ = agent.test_env.step(agent.act(state))
            episode_return += reward
            episode_length += 1
        if not agent.demo:
            agent.metrics[agent.metric_dict["reward"]].add(episode_return)
            agent.metrics[agent.metric_dict["length"]].add(episode_length)
    if agent.demo:
        agent.env.close()
        agent.test_env.close()