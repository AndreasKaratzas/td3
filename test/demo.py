
import sys
sys.path.append('../')

import gymnasium as gym

from gymnasium.wrappers import Monitor

from src.test import test
from lib.agent import Agent
from src.args import arguments
from utils.logger import HardLogger
from src.model import MLPActorCritic
from utils.msg import info, print_test_message
from utils.functions import parse_configs, update_args


if __name__ == '__main__':
    # parse arguments
    args = arguments()
    if args.info:
        info()

    if args.config:
        settings = parse_configs(filepath=args.config)
        args = update_args(args, settings)
        args.export_configs = False
    else:
        args.export_configs = True

    logger = HardLogger(
        output_dir=args.checkpoint_dir,
        output_fname=args.logger_name,
        exp_name=args.name,
        demo=True
    )

    print_test_message(
        agent="TD3 with " + ("Priority Experience Replay" if args.buffer_arch ==
                             'priority' else "Random Experience Replay") + " and " + args.arch.upper() + " core",
        env_id=args.env, epochs=args.demo_episodes, device=args.device,
        parent_dir_printable_version=logger.parent_dir_printable_version)

    # create RL environment
    env = gym.make(args.env)
    env = Monitor(env, logger.demo_dir, force=True)

    # create the TD3 agent
    agent = Agent(env=env, env_id=args.env, actor_critic=MLPActorCritic, arch=args.arch, activation=args.activation,
                  seed=args.seed, prior_eps=args.prior_eps, learning_starts=args.learning_starts, beta=args.beta,
                  epochs=args.epochs, replay_size=args.replay_size, gamma=args.gamma, gradient_steps=args.gradient_steps,
                  polyak=args.polyak, auto_save=args.auto_save, elite_criterion=args.elite_criterion, name=args.name,
                  lr_actor=args.lr_actor, lr_critic=args.lr_critic, batch_size=args.batch_size, alpha=args.alpha,
                  demo_episodes=args.demo_episodes, max_ep_len=args.max_ep_len, logger=logger, hidden_sizes=args.hidden_sizes,
                  checkpoint_freq=args.checkpoint_freq, debug_mode=args.debug_mode, checkpoint_dir=logger.model_dir,
                  device=args.device, export_configs=args.export_configs, load_checkpoint=args.load_checkpoint,
                  mu=args.mu, sigma=args.sigma, noise_dist=args.noise_dist, theta=args.theta, buffer_arch=args.buffer_arch,
                  target_noise=args.target_noise, noise_clip=args.noise_clip, policy_delay=args.policy_delay,
                  steps_per_epoch=args.steps_per_epoch, update_every=args.update_every, demo=True)

    agent.load(agent_checkpoint_path=agent.load_checkpoint)

    test(agent=agent, demo_episodes=args.demo_episodes)
    
    env.close()
