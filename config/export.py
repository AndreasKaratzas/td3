
import sys
sys.path.append('../')


def build_experiment_configs(agent):
    return {
        'experiment':
            {
                'alias': agent.name,
                'logger': str(agent.logger.log_f_name.resolve())
            },
        'env':
            {
                'name': agent.env_id,
                'max_ep_len': agent.max_ep_len
            },
        'td3':
            {
                'extractor': str(agent.hidden_sizes),
                'arch': agent.extractor,
                'activation': agent.activation,
                'pi_lr': agent.lr_actor,
                'q_lr': agent.lr_critic,
                'replay_size': agent.replay_size,
                'polyak': agent.polyak,
                'gamma': agent.gamma,
                'prior_eps': agent.prior_eps,
                'learning_starts': agent.learning_starts,
                'gradient_steps': agent.gradient_steps,
                'steps_per_epoch': agent.steps_per_epoch,
                'update_every': agent.update_every,
                'policy_delay': agent.policy_delay,
                'target_noise_clip': agent.target_noise_clip,
                'target_policy_noise': agent.target_policy_noise
            },
        'training':
            {
                'epochs': agent.epochs,
                'batch_size': agent.batch_size,
                'demo_episodes': agent.demo_episodes
            },
        'exploration':
            {
                'mu': agent.mu,
                'sigma': agent.sigma,
                'noise_dist': agent.noise_dist,
                'theta': agent.theta
            },
        'per':
            {
                'buffer_arch': agent.buffer_arch,
                'beta': agent.beta,
                'alpha': agent.alpha
            },
        'auxiliary':
            {
                'checkpoint_freq': agent.checkpoint_freq,
                'seed': agent.seed,
                'checkpoint_dir': str(agent.checkpoint_dir.resolve()),
                'device': agent.device.type,
                'elite_metric': agent.elite_criterion,
                'auto_save': agent.auto_save
            }
    }
