
import sys
sys.path.append('../')

import time
import torch
import psutil
import numpy as np
import torcheval.metrics as tnt

from src.test import test
from lib.agent import Agent
from utils.plot import compile_plots
from utils.logger import ProgressStatus
from common.scaler import scale_action, unscale_action
from config.settings import FORMAT, NAMES_DICT, OPERATORS


def train(agent: Agent):

    # initialize logger placeholders
    epoch_time = tnt.Mean()
    report = ProgressStatus(format=FORMAT, names=NAMES_DICT, operators=OPERATORS)
    ep_start = time.time()
    init_msg = f"{'Epoch':>9}{'epoch_time':>13}{'gpu_mem':>9}{'ram_util':>9}{'avg_length':>12}{'avg_reward':>12}{'avg_q_val':>12}{'loss_actor':>12}{'loss_critic':>12}"
    print("\n\n" + init_msg)
    agent.logger.log_message(init_msg)

    # Prepare for interaction with environment
    state, episode_return, episode_length = agent.env.reset(seed=agent.seed), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for timestep in range(agent.total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy (with some noise).
        if timestep > agent.learning_starts:
            action = agent.act(state)
        else:
            action = agent.env.action_space.sample()

        scaled_action = scale_action(
            action_space=agent.env.action_space, action=action)
        scaled_action = np.clip(scaled_action + agent.noise_fn(), -1, 1)

        # We store the scaled action in the buffer
        buffer_action = scaled_action
        action = unscale_action(
            action_space=agent.env.action_space, action=scaled_action)

        # Step the env
        next_state, reward, done, truncated, info = agent.env.step(action)
        episode_return += reward
        episode_length += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        done = False if episode_length == agent.max_ep_len else done

        # Store experience to replay buffer
        agent.cache(state, next_state, buffer_action, reward, done)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        state = next_state

        # End of trajectory handling
        if done or (episode_length == agent.max_ep_len):
            # print(f"Training episode just finished in timestep {episode_length} with reward {episode_return}")
            state, episode_return, episode_length = agent.env.reset(seed=agent.seed), 0, 0
            agent.noise_fn.reset()

        # Update handling
        if timestep >= agent.learning_starts and timestep % agent.update_every == 0:
            for grad_step in range(agent.gradient_steps):
                agent.learn(timestep=timestep)

        # End of epoch handling
        if (timestep+1) % agent.steps_per_epoch == 0:
            epoch = (timestep+1) // agent.steps_per_epoch

            # Test the performance of the deterministic version of the agent.
            test(agent)

            # Update benchmark metric
            agent.epoch = epoch
            ep_end = time.time()
            epoch_time.update(ep_end - ep_start)

            # Update progress metrics
            for metric in agent.metrics:
                metric.update()

            # Log agent progress
            msg = report.compile(metrics={
                "epoch": epoch,
                "epochs": agent.epochs,
                "bench": round(epoch_time.compute(), 3),
                "cuda_mem": round(torch.cuda.memory_reserved() /
                                  1E6, 3) if agent.device.type == 'cuda' else 0,
                "ram_util": psutil.virtual_memory().percent,
                "length": np.ceil(agent.metrics[agent.metric_dict["length"]].avg).astype(int),
                "reward": agent.metrics[agent.metric_dict["reward"]].avg,
                "q_val": agent.metrics[agent.metric_dict["q_val"]].avg,
                "loss_pi": agent.metrics[agent.metric_dict["loss_pi"]].avg,
                "loss_q": agent.metrics[agent.metric_dict["loss_q"]].avg
            }, show_cuda=agent.device.type == 'cuda')

            # Save snapshot of agent
            agent.store()

            # Reset all metrics
            for metric in agent.metrics:
                metric.reset()

            # Log progress
            agent.logger.log_message(msg)
            ep_start = time.time()

    compile_plots(log_f_name=agent.logger.log_f_name, plot_dir=agent.logger.plot_dir)
