
import sys
sys.path.append('../')

import itertools
import numpy as np
import torch
import time

from copy import deepcopy
from torch.optim import Adam

from src.utils import polyak_update
from src.replay import ReplayBuffer
from src.priority import PrioritizedReplayBuffer
from src.model import MLPActorCritic
from src.loss import weighted_mse_loss
from src.scaler import scale_action, unscale_action
from src.noise import OrnsteinUhlenbeckActionNoise


class Agent:
    def __init__(
        self,
        env,
        sigma=0.5,
        theta=0.15, 
        device='cuda:0',
        batch_size=128,
        replay_size=1_000_000,
        noise_clip=0.5,
        act_noise=0.1,
        epochs=20,
        max_ep_len=1000,
        start_steps=1000,
        gradient_steps=10,
        polyak=0.001,
        lr_actor=0.001,
        lr_critic=0.001,
        gamma=0.99,
        policy_delay=2,
        alpha=0.2,
        prior_eps=1e-6,
        beta=0.6
    ):

        self.env = deepcopy(env)
        self.sigma = sigma
        self.theta = theta
        self.tau = polyak
        self.beta = beta
        self.gamma = gamma
        self.prior_eps = prior_eps
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.policy_delay = policy_delay

        self.target_policy_noise = act_noise
        self.target_noise_clip = noise_clip
        self.num_timesteps = max_ep_len * (epochs - 1)
        self.learning_starts = start_steps
        self.gradient_steps = gradient_steps

        # env dims
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape
        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = self.env.action_space.high[0]

        self.device = torch.device('cuda:0' if torch.cuda.is_available(
        ) else 'cpu') if device is None else torch.device(device=device)

        self.ounoise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(self.env.action_space.shape[-1]), sigma=float(self.sigma) * np.ones(self.env.action_space.shape[-1]), theta=self.theta)
        
        self.buffer = PrioritizedReplayBuffer(
                obs_dim=self.obs_dim, size=replay_size, act_dim=self.act_dim, batch_size=batch_size, alpha=alpha)

        # Create actor-critic module and target networks
        self.online = MLPActorCritic(self.env.observation_space, self.env.action_space)
        self.target = deepcopy(self.online)

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
        a = self.online.act(torch.as_tensor(state, dtype=torch.float32).to(self.device))
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

        samples = self.buffer.sample_batch(self.beta)

        state = torch.FloatTensor(samples["obs"]).to(self.device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(self.device)
        action = torch.FloatTensor(samples["acts"]).to(self.device)
        reward = torch.FloatTensor(samples["rews"]).to(self.device)
        done = torch.FloatTensor(samples["done"]).to(self.device)
        weights = torch.FloatTensor(samples["weights"]).to(self.device)
        indices = samples["indices"]

        return state, action, reward, next_state, done, weights, indices

    def train(self, timestep):
        
        actor_losses, critic_losses = [], []

        for step in range(self.gradient_steps):
            # Sample replay buffer
            state, action, reward, next_state, done, weights, indices = self.recall()

            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = action.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip,
                                    self.target_noise_clip)
                next_actions = (self.target.actor(
                    next_state) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_a = self.target.critic_a(next_state, next_actions)
                next_q_b = self.target.critic_b(next_state, next_actions)
                next_q_values = torch.min(next_q_a, next_q_b)
                target_q_values = reward + \
                    (1 - done) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_a = self.online.critic_a(state, action)
            current_q_b = self.online.critic_b(state, action)
            
            # loss_q_a = F.mse_loss(current_q_a, target_q_values)
            # loss_q_b = F.mse_loss(current_q_b, target_q_values)
            
            loss_q_a = weighted_mse_loss(
                input=current_q_a, 
                target=target_q_values, 
                weight=weights
            )
            
            loss_q_b = weighted_mse_loss(
                input=current_q_b, 
                target=target_q_values, 
                weight=weights
            )

            # Compute critic loss
            critic_loss = loss_q_a + loss_q_b
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Useful info for logging
            td_error_a = current_q_a - target_q_values
            td_error_b = current_q_b - target_q_values
            td_error = torch.cat((td_error_a.view(
                td_error_a.shape[-1], 1), td_error_b.view(td_error_b.shape[-1], 1)), dim=1)
            td_error = torch.mean(td_error, dim=1, keepdim=True).squeeze(1)

            # PER: update priorities
            loss_for_prior = td_error.detach().cpu().numpy()
            new_priorities = np.abs(loss_for_prior) + self.prior_eps
            self.buffer.update_priorities(indices, new_priorities)

            # PER: increase beta
            fraction = min(timestep / self.num_timesteps, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            if timestep % self.policy_delay:

                # Freeze Q-network so you don't waste computational effort
                # computing gradients for it during the policy learning step.
                for parameter in self.critic_params:
                    parameter.requires_grad = False

                # Compute actor loss
                actor_loss = -self.online.critic_a(
                    state, self.online.actor(state)).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Unfreeze Q-network so you can optimize it at next TD3 step.
                for parameter in self.critic_params:
                    parameter.requires_grad = True

                self.actor_loss = actor_loss

            polyak_update(self.online.parameters(),
                            self.target.parameters(), self.tau)

    def learn(self):
        timesteps = 0
        state, episode_return, episode_length = self.env.reset(), 0, 0
        start = time.time()
        while timesteps < self.num_timesteps:
            state, episode_return, episode_length = self.collect_rollouts(
                timesteps, state, episode_return, episode_length)
            if timesteps > self.learning_starts:
                self.train(timesteps)
                # self.test(demo_episodes=5)
            timesteps += 1
            bench = time.time() - start
            # print(f"Timestep: {timesteps} with benchmark {bench}")
            start = time.time()
    
    def test(self, demo_episodes):
        demo_episode_return, demo_episode_length = [], []
        for demo in range(demo_episodes):
            state, done, episode_return, episode_length = self.env.reset(), False, 0, 0
            while not(done or (episode_length == 1000)):
                action = self.act(state)
                state, reward, done, info = self.env.step(action)
                episode_return += reward
                episode_length += 1
            demo_episode_length.append(episode_length)
            demo_episode_return.append(episode_return)

        print(
            f"Mean reward: {np.mean(demo_episode_return)}\tMean length: {np.mean(demo_episode_length)}")

    def _sample_action(self, timestep, state):
        # Select action randomly or according to policy
        if timestep < self.learning_starts:
            # Warmup phase
            unscaled_action = self.env.action_space.sample()
        else:
            unscaled_action = self.act(state)
        
        scaled_action = scale_action(action_space=self.env.action_space, action=unscaled_action)
        scaled_action = np.clip(scaled_action + self.ounoise(), -1, 1)

        # We store the scaled action in the buffer
        buffer_action = scaled_action
        action = unscale_action(action_space=self.env.action_space, action=scaled_action)
        
        return action, buffer_action

    def collect_rollouts(self, timestep, state, episode_return, episode_length):

        # Select action randomly or according to policy
        actions, buffer_actions = self._sample_action(
            timestep, state)

        # Rescale and perform action
        new_obs, rewards, done, _ = self.env.step(actions)

        episode_return += rewards
        episode_length += 1
        
        self.cache(state, new_obs, buffer_actions, rewards, done)

        if done:
            print(f"An episode just finished in timestep {episode_length} with total reward {episode_return}.")
            episode_length = 0
            episode_return = 0
            new_obs = self.env.reset()
            self.ounoise.reset()
        
        return new_obs, episode_return, episode_length
