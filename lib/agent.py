
import sys
sys.path.append('../')

import os
import gymnasium as gym
import torch
import itertools
import numpy as np

from pathlib import Path
from copy import deepcopy
from torch.optim import Adam

from utils.metric import Metric
from utils.logger import HardLogger
from common.replay import ReplayBuffer
from config.settings import NAMES, OPERATORS_STR
from src.model import MLPActorCritic, count_vars
from common.priority import PrioritizedReplayBuffer
from common.loss import weighted_mse_loss, mse_loss
from config.export import build_experiment_configs
from utils.functions import _seed, colorstr, str2activation, polyak_update
from common.noise import UniformActionNoise, NormalActionNoise, OrnsteinUhlenbeckActionNoise


class Agent:
    def __init__(self, env: gym.Env, env_id: str, logger: HardLogger, actor_critic=MLPActorCritic, 
                 replay_size=int(1e6), gamma=0.99, epochs=100, export_configs=False, checkpoint_dir='.', device='cpu',
                 polyak=0.995, learning_starts=1000, lr_actor=1e-3, lr_critic=1e-3, batch_size=128, auto_save=True,
                 gradient_steps=10, checkpoint_freq=1, name='exp', alpha=0.4, theta=0.15, buffer_arch='random',
                 max_ep_len=1000, demo_episodes=10, load_checkpoint='model.pth', beta=0.6, elite_criterion='avg_q_val',
                 debug_mode=False, seed=0, prior_eps=1e-6, mu=0.0, sigma=0.1, noise_dist='uniform', update_every=10,
                 target_noise=0.2, noise_clip=0.5, policy_delay=2, arch='mlp', activation='relu', hidden_sizes=[256, 256],
                 steps_per_epoch=2500, demo=False):

        # initialize envs
        self.env_id = env_id
        self.env = deepcopy(env)
        self.test_env = deepcopy(env)

        # initialize logger
        self.logger = logger

        # noise hyper parameters
        self.mu = mu
        self.sigma = sigma
        self.theta = theta

        # priority replay hyper parameters
        self.beta = beta
        self.alpha = alpha
        self.prior_eps = prior_eps
        self.noise_dist = noise_dist.lower()

        # auxiliary hyper parameters
        self.demo = demo
        self.seed = seed
        self.name = name
        self.auto_save = auto_save
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        self.elite_criterion = elite_criterion
        self.load_checkpoint = load_checkpoint

        # helper variables
        self.epoch = 0
        self.timestep = 0

        # agent hyper parameters
        self.gamma = gamma
        self.epochs = epochs
        self.polyak = polyak
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.batch_size = batch_size
        self.max_ep_len = max_ep_len
        self.buffer_arch = buffer_arch
        self.update_every = update_every
        self.demo_episodes = demo_episodes
        self.gradient_steps = gradient_steps
        self.learning_starts = learning_starts
        self.steps_per_epoch = steps_per_epoch
        
        # model hyper parameters
        self.arch = arch
        self.activation = activation
        self.hidden_sizes = hidden_sizes

        # td3 specific hyper parameters
        self.policy_delay = policy_delay
        self.target_noise_clip = noise_clip
        self.target_policy_noise = target_noise

        # utilize a training device
        self.device = torch.device('cuda:0' if torch.cuda.is_available(
        ) else 'cpu') if device is None else torch.device(device=device)

        # seed random number generators for debugging purposes
        if debug_mode:
            _seed(env=self.env, device=self.device, seed=seed)

        # env dims
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape

        # set max episode length
        self.max_ep_len = max_ep_len
        self.total_steps = self.steps_per_epoch * (epochs - 1)

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = self.env.action_space.high[0]
        assert self.target_noise_clip <= self.act_limit

        # Create actor-critic module and target networks
        self.online = actor_critic(
            observation_space=self.env.observation_space, 
            action_space=self.env.action_space,
            hidden_sizes=self.hidden_sizes,
            activation=str2activation(self.activation),
            arch=self.arch
        )
        self.target = deepcopy(self.online)

        # create a placeholder with both `critic_a` and `critic_b` parameters for easy referencing 
        self.critic_params = itertools.chain(self.online.critic_a.parameters(), self.online.critic_b.parameters())

        # sync target with online
        self.target.actor.load_state_dict(
            self.online.actor.state_dict())
        self.target.critic_a.load_state_dict(
            self.online.critic_a.state_dict())
        self.target.critic_b.load_state_dict(
            self.online.critic_b.state_dict())

        # upload models to device
        self.online.actor.to(self.device)
        self.online.critic_a.to(self.device)
        self.online.critic_b.to(self.device)
        self.target.actor.to(self.device)
        self.target.critic_a.to(self.device)
        self.target.critic_b.to(self.device)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for parameter in self.target.parameters():
            parameter.requires_grad = False
        
        # Set up optimizers for policy and q-function
        self.actor_optimizer = Adam(self.online.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = Adam(self.critic_params, lr=self.lr_critic)

        # set up replay buffer size
        exponent = np.ceil(np.log2(replay_size))
        new_replay_size = 2 ** int(exponent)

        if new_replay_size != replay_size:
            replay_size = new_replay_size
            self.logger.log_message(f"Resetting `replay_size` to {replay_size}" +
                                    f" to be a power of 2 for memory purposes.")
            print(f"\n\t\t           Resetting `{colorstr(options=['red', 'underline'], string_args=['replay_size'])}` to {colorstr(options=['green', 'bold'], string_args=[str(replay_size)])}" +
                  f"\n\t\t          to be a power of 2 for memory purposes.")
        else:
            self.logger.log_message(f"Setting `replay_size` to {replay_size}.")

        # create experience replay buffer 
        assert buffer_arch in ['random', 'priority']
        if self.buffer_arch == 'random':
            self.buffer = ReplayBuffer(
                obs_dim=self.obs_dim, size=replay_size, act_dim=self.act_dim, batch_size=batch_size
            )
        if self.buffer_arch == 'priority':
            # Prioritized Experience Replay
            self.buffer = PrioritizedReplayBuffer(
                obs_dim=self.obs_dim, size=replay_size, act_dim=self.act_dim, batch_size=batch_size, alpha=alpha)

        # Count variables
        var_counts = tuple(count_vars(module) for module in [
                           self.online.actor, self.online.critic_a, self.online.critic_b])
        self.logger.log_message(
            '\nNumber of parameters: \t actor: %d, \t critic_a: %d, \t critic_b: %d\n' % var_counts)
        
        # complete agent setup
        self._setup()
        self.replay_size = replay_size

        # save the experiment configuration
        if export_configs and not demo:
            self.config = build_experiment_configs(self)
            self.logger.export_yaml(d=self.config, filename=name)

    def _setup(self):
        selection_name = ''
        if len(self.elite_criterion.split('_')) > 1:
            selection_name = '_'.join(self.elite_criterion.split('_')[1:])

        # setup progress metrics
        self.chkpt_cntr = 0
        self.metrics = [Metric(name=name, selection_metric='avg', comp_operator=op, auto_update=False)
                        for name, op in zip(NAMES, OPERATORS_STR)]
        self.metric_dict = {name: idx for idx, name in enumerate(NAMES)}
        self.selection_name = selection_name
        
        # setup noise function
        assert self.noise_dist in ['gaussian', 'uniform', 'ounoise']
        
        if self.noise_dist == 'gaussian':
            self.noise_fn = NormalActionNoise(
                mean=float(self.mu) * np.ones(self.env.action_space.shape[-1]),
                sigma=float(self.sigma) *
                np.ones(self.env.action_space.shape[-1])
            )
        
        if self.noise_dist == 'ounoise':
            self.noise_fn = OrnsteinUhlenbeckActionNoise(
                mean=float(self.mu) * np.ones(self.env.action_space.shape[-1]),
                sigma=float(self.sigma) *
                np.ones(self.env.action_space.shape[-1]),
                theta=self.theta
            )
        
        if self.noise_dist == 'uniform':
            self.noise_fn = UniformActionNoise(
                low=-self.act_limit,
                high=self.act_limit,
                size=self.act_dim[0]
            )
    
    def act(self, state):
        """Given a state, choose an action and update value of step.
        Parameters
        ----------
        state : np.ndarray
            A single observation of the current state.
        Returns
        -------
        int
            A float representing the environment action.
        """
        a = self.online.act(torch.as_tensor(
            state, dtype=torch.float32).to(self.device))
        return a

    def cache(self, state, next_state, action, reward, done):
        """Stores the experience replay and priority buffers.
        Parameters
        ----------
        state : numpy.ndarray
            The state of the agent at a time step `t`.
        next_state : numpy.ndarray
            The state of the agent at the next time step `t + 1`.
        action : int
            The action selected by the agent at a time step `t`.
        reward : float
            The reward accumulated by the agent at a time step `t`.
        done : bool
            The terminal indicator at a time step `t`.
        """

        Transition = [state, action, reward, next_state, done]

        # Add a single step transition
        self.buffer.store(*Transition)

    def recall(self):
        """Retrieve a batch of experiences from the experience replay.
        Returns
        -------
        Tuple
            A batch of experiences fetched by the experience replay.
        """

        samples = self.buffer.sample_batch(
        ) if self.buffer_arch == 'random' else self.buffer.sample_batch(self.beta)

        state = torch.FloatTensor(samples["obs"]).to(self.device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(self.device)
        action = torch.FloatTensor(samples["acts"]).to(self.device)
        reward = torch.FloatTensor(samples["rews"]).to(self.device)
        done = torch.FloatTensor(samples["done"]).to(self.device)

        weights = None
        indices = None
        if self.buffer_arch == 'priority':
            weights = torch.FloatTensor(samples["weights"]).to(self.device)
            indices = samples["indices"]

        return state, action, reward, next_state, done, weights, indices

    # Set up function for computing TD3 Q-losses
    def compute_loss_q(self, state, action, reward, next_state, done, weights):
        
        # Get current Q-values estimates for each critic network
        current_critic_a = self.online.critic_a(state, action)
        current_critic_b = self.online.critic_b(state, action)

        # Bellman backup for Q functions
        with torch.no_grad():
            next_actions = self.target.actor(next_state)

            # Target policy smoothing
            noise = torch.randn_like(next_actions) * self.target_policy_noise
            noise = torch.clamp(
                noise, -self.target_noise_clip, self.target_noise_clip)
            next_actions = next_actions + noise
            next_actions = torch.clamp(next_actions, -self.act_limit, self.act_limit)

            # Target Q-values
            next_critic_a = self.target.critic_a(next_state, next_actions)
            next_critic_b = self.target.critic_b(next_state, next_actions)
            next_critic_values = torch.min(next_critic_a, next_critic_b)
            backup = reward + self.gamma * (1 - done) * next_critic_values
        
        # MSE loss against Bellman backup
        if self.buffer_arch == 'priority':
            loss_critic_a = weighted_mse_loss(
                input=current_critic_a,
                target=backup,
                weight=weights
            )

            loss_critic_b = weighted_mse_loss(
                input=current_critic_b,
                target=backup,
                weight=weights
            )

        if self.buffer_arch == 'random':
            loss_critic_a = mse_loss(
                input=current_critic_a,
                target=backup
            )
            loss_critic_b = mse_loss(
                input=current_critic_b,
                target=backup
            )

        # critic loss
        loss_critic = loss_critic_a + loss_critic_b

        # for priority replay
        td_error_a = current_critic_a - backup
        td_error_b = current_critic_b - backup
        td_error = torch.cat((td_error_a.view(
            td_error_a.shape[-1], 1), td_error_b.view(td_error_b.shape[-1], 1)), dim=1)
        td_error = torch.mean(td_error, dim=1, keepdim=True).squeeze(1)

        # Useful info for logging
        q_val = torch.cat((current_critic_a.view(
            current_critic_a.shape[-1], 1), current_critic_b.view(current_critic_b.shape[-1], 1)), dim=1)
        q_val = torch.mean(
            q_val, dim=1, keepdim=True).detach().cpu().numpy().squeeze(1)
    
        return loss_critic, td_error, np.mean(q_val)

    # Set up function for computing TD3 pi loss
    def compute_loss_pi(self, state):
        return -self.online.critic_a(
            state, self.online.actor(state)).mean()

    def update(self, timestep, state, action, reward, next_state, done, weights):
        # First run one gradient descent step for Q1 and Q2
        self.critic_optimizer.zero_grad()
        critic_loss, td_error, q_val = self.compute_loss_q(
            state, action, reward, next_state, done, weights)
        critic_loss.backward()
        self.critic_optimizer.step()
        self.metrics[self.metric_dict["loss_q"]].add(critic_loss.item())

        # Possibly update pi and target networks
        if timestep % self.policy_delay == 0:

            # Freeze Q-networks so you don't waste computational effort
            # computing gradients for them during the policy learning step.
            for parameter in self.critic_params:
                parameter.requires_grad = False

            # Next run one gradient descent step for pi.
            self.actor_optimizer.zero_grad()
            actor_loss = self.compute_loss_pi(state)
            actor_loss.backward()
            self.actor_optimizer.step()
            self.metrics[self.metric_dict["loss_pi"]].add(actor_loss.item())

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for parameter in self.critic_params:
                parameter.requires_grad = True

            # Finally, update target networks by polyak averaging.
            polyak_update(self.online.parameters(),
                          self.target.parameters(), self.polyak)
        
        return td_error, q_val
    
    def learn(self, timestep):
        if len(self.buffer) < self.batch_size:
            return None
        
        # Sample from memory
        state, action, reward, next_state, done, weights, indices = \
            self.recall()
        
        td_error, q_val = self.update(
            timestep, state, action, reward, next_state, done, weights)
        
        self.metrics[self.metric_dict["q_val"]].add(q_val)

        if self.buffer_arch == 'priority':
            # PER: update priorities
            loss_for_prior = td_error.detach().cpu().numpy()
            new_priorities = np.abs(loss_for_prior) + self.prior_eps
            self.buffer.update_priorities(indices, new_priorities)

            # PER: increase beta
            fraction = min(self.epoch / self.epochs, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

    def elite_criterion_factory(self) -> bool:
        if not self.auto_save:
            if self.chkpt_cntr % self.checkpoint_freq:
                return True
            self.chkpt_cntr += 1
        
        checkpoint_flag = self.metrics[self.metric_dict[self.selection_name]].status(
            status_reset=False)
        
        return checkpoint_flag

    def load(self, agent_checkpoint_path):
        agent_checkpoint_path = Path(agent_checkpoint_path)
        if not agent_checkpoint_path.exists():
            raise ValueError(f"{agent_checkpoint_path} does not exist")

        ckp = torch.load(agent_checkpoint_path, map_location=self.device)

        print(f"Loading model at {agent_checkpoint_path}")

        self.online.actor.load_state_dict(ckp.get('online_actor'))
        self.online.critic_a.load_state_dict(ckp.get('online_critic_a'))
        self.online.critic_b.load_state_dict(ckp.get('online_critic_b'))
        self.target.actor.load_state_dict(ckp.get('target_actor'))
        self.target.critic_a.load_state_dict(ckp.get('target_critic_a'))
        self.target.critic_b.load_state_dict(ckp.get('target_critic_b'))
        self.actor_optimizer.load_state_dict(ckp.get('actor_optimizer'))
        self.critic_optimizer.load_state_dict(ckp.get('critic_optimizer'))

        self.metrics[self.metric_dict["length"]].reg = ckp.get('_length')
        self.metrics[self.metric_dict["reward"]].reg = ckp.get('_reward')
        self.metrics[self.metric_dict["q_val"]].reg = ckp.get('_q_val')
        self.metrics[self.metric_dict["loss_pi"]].reg = ckp.get('_loss_actor')
        self.metrics[self.metric_dict["loss_q"]].reg = ckp.get('_loss_critic')

        # Sync progress metrics
        for metric in self.metrics:
            metric.update()

        print(
            f"Loaded checkpoint with:"
            f"\n\t * {self.metrics[self.metric_dict['reward']].avg:7.3f} mean accumulated test reward"
            f"\n\t * {self.metrics[self.metric_dict['length']].avg:7.3f} mean test episode length"
            f"\n\t * {self.metrics[self.metric_dict['q_val']].avg:7.3f} mean Q value achieved"
            f"\n\t * {self.metrics[self.metric_dict['loss_pi']].avg:7.3f} mean actor model loss"
            f"\n\t * {self.metrics[self.metric_dict['loss_q']].avg:7.3f} mean critic model loss")

    def store(self):

        if self.elite_criterion_factory():
            torch.save({
                'online_actor': self.online.actor.state_dict(),
                'online_critic_a': self.online.critic_a.state_dict(),
                'online_critic_b': self.online.critic_b.state_dict(),
                'target_actor': self.target.actor.state_dict(),
                'target_critic_a': self.target.critic_a.state_dict(),
                'target_critic_b': self.target.critic_b.state_dict(),
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
                '_reward': self.metrics[self.metric_dict['reward']].reg,
                '_length': self.metrics[self.metric_dict['length']].reg,
                '_q_val': self.metrics[self.metric_dict['q_val']].reg,
                '_loss_actor': self.metrics[self.metric_dict['loss_pi']].reg,
                '_loss_critic': self.metrics[self.metric_dict['loss_q']].reg
            }, os.path.join(
                self.checkpoint_dir,
                f"epoch_{self.epoch:05d}-" + 
                f"avg_reward{self.metrics[self.metric_dict['reward']].avg:07.3f}-" +
                f"ep_length{self.metrics[self.metric_dict['length']].avg:07.3f}-" +
                f"avg_q_val{self.metrics[self.metric_dict['q_val']].avg:07.3f}-" + 
                f"loss_actor_{self.metrics[self.metric_dict['loss_pi']].avg:07.3f}-" +
                f"loss_critic_{self.metrics[self.metric_dict['loss_q']].avg:07.3f}.pth"
            ))
