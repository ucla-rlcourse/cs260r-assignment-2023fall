import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .buffer import PPORolloutStorage
from .network import PPOModel


class PPOConfig:
    """Not like previous assignment where we use a dict as config, here we
    build a class to represent config."""

    def __init__(self):
        # Common
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_freq = 5
        self.log_freq = 1
        self.num_processes = 10

        # Sample
        self.num_steps = 4_000  # num_steps * num_processes = sample_batch_size

        # Learning
        self.gamma = 0.99
        self.lr = 1e-5
        self.ppo_epoch = 10
        self.mini_batch_size = 256
        self.ppo_clip_param = 0.2
        self.use_gae = True
        self.gae_lambda = 0.95
        self.entropy_loss_weight = 0.0
        self.value_loss_weight = 0.5
        self.grad_norm_max = 0.5


class PPOTrainer:
    def __init__(self, config, num_features=161, num_actions=2):
        self.device = config.device
        self.config = config
        self.lr = config.lr
        self.num_processes = config.num_processes
        self.gamma = config.gamma
        self.num_steps = config.num_steps

        self.grad_norm_max = config.grad_norm_max
        self.value_loss_weight = config.value_loss_weight
        self.entropy_loss_weight = config.entropy_loss_weight

        self.num_actions = num_actions
        self.num_features = num_features

        self.setup_model_and_optimizer()

        self.act_dim = self.num_actions

        self.rollouts = PPORolloutStorage(
            self.num_steps, self.num_processes, self.act_dim, self.num_features, self.device, False,
            self.config.use_gae, self.config.gae_lambda
        )

        # There configs are only used in PPO
        self.ppo_epoch = config.ppo_epoch
        self.mini_batch_size = config.mini_batch_size
        self.clip_param = config.ppo_clip_param

    def setup_model_and_optimizer(self):
        self.model = PPOModel(self.num_features, self.num_actions, False)
        self.model = self.model.to(self.device)
        self.model.train()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def process_obs(self, obs):
        # Change to tensor, change type, add batch dimension for observation.
        if not isinstance(obs, torch.Tensor):
            obs = np.asarray(obs)
            obs = torch.from_numpy(obs.astype(np.float32)).to(self.device)
        obs = obs.float()
        if obs.ndim == 1 or obs.ndim == 3:  # Add additional batch dimension.
            obs = obs.view(1, *obs.shape)
        return obs

    def compute_action(self, obs, deterministic=False):
        obs = self.process_obs(obs)
        means, log_std, values = self.model(obs)
        action_std = torch.exp(log_std)
        dist = torch.distributions.Normal(means, action_std)
        if deterministic:  # Use the means as the action
            actions = means
        else:
            actions = dist.sample()
        action_log_probs = dist.log_prob(actions).sum(dim=-1)
        actions = actions.view(-1, self.num_actions)
        values = values.view(-1, 1)
        action_log_probs = action_log_probs.view(-1, 1)
        return values, actions, action_log_probs

    def evaluate_actions(self, obs, act):
        """Run models to get the values, log probability and action
        distribution entropy of the action in current state"""
        obs = self.process_obs(obs)
        assert torch.is_floating_point(act)
        means, log_std, values = self.model(obs)
        action_std = torch.exp(log_std)
        dist = torch.distributions.Normal(means, action_std)
        action_log_probs_raw = dist.log_prob(act)
        action_log_probs = action_log_probs_raw.sum(dim=-1)
        dist_entropy = dist.entropy().sum(-1)
        values = values.view(-1, 1)
        action_log_probs = action_log_probs.view(-1, 1)
        return values, action_log_probs, dist_entropy

    def compute_values(self, obs):
        """Compute the values corresponding to current policy at current
        state"""
        obs = self.process_obs(obs)
        _, _, values = self.model(obs)
        return values

    def save_w(self, log_dir="", suffix=""):
        os.makedirs(log_dir, exist_ok=True)
        save_path = os.path.join(log_dir, "checkpoint-{}.pkl".format(suffix))
        torch.save(dict(
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict()
        ), save_path)
        return save_path

    def load_w(self, log_dir="", suffix=""):
        log_dir = os.path.abspath(os.path.expanduser(log_dir))
        save_path = os.path.join(log_dir, "checkpoint-{}.pkl".format(suffix))
        if os.path.isfile(save_path):
            state_dict = torch.load(
                save_path,
                torch.device('cpu') if not torch.cuda.is_available() else None
            )
            self.model.load_state_dict(state_dict["model"])
            self.optimizer.load_state_dict(state_dict["optimizer"])
            print("Successfully loaded weights from {}!".format(save_path))
            return True
        else:
            raise ValueError("Failed to load weights from {}! File does not exist!".format(save_path))

    def compute_loss(self, sample):
        """Compute the loss of PPO"""
        observations_batch, actions_batch, value_preds_batch, return_batch, masks_batch, \
        old_action_log_probs_batch, adv_targ = sample

        assert old_action_log_probs_batch.shape == (self.mini_batch_size, 1)
        assert adv_targ.shape == (self.mini_batch_size, 1)
        assert return_batch.shape == (self.mini_batch_size, 1)

        values, action_log_probs, dist_entropy = self.evaluate_actions(observations_batch, actions_batch)

        assert values.shape == (self.mini_batch_size, 1)
        assert action_log_probs.shape == (self.mini_batch_size, 1)
        assert values.requires_grad
        assert action_log_probs.requires_grad
        assert dist_entropy.requires_grad

        # Implement policy loss
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        policy_loss = -torch.min(surr1, surr2)
        policy_loss_mean = policy_loss.mean()

        # Implement value loss
        value_loss = F.mse_loss(input=values, target=return_batch)
        value_loss_mean = value_loss.mean()

        # This is the total loss
        loss = policy_loss + self.config.value_loss_weight * value_loss - self.config.entropy_loss_weight * dist_entropy
        loss = loss.mean()

        return loss, policy_loss_mean, value_loss_mean, torch.mean(dist_entropy), torch.mean(ratio)

    def update(self, rollout):
        # Get the normalized advantages
        advantages = rollout.returns[:-1] - rollout.value_preds[:-1]
        adv_mean = advantages.mean().item()
        advantages = (advantages - advantages.mean()) / max(advantages.std(), 1e-4)

        value_loss_epoch = []
        policy_loss_epoch = []
        dist_entropy_epoch = []
        total_loss_epoch = []
        norm_epoch = []
        ratio_epoch = []

        assert advantages.shape[0] * advantages.shape[
            1] >= self.mini_batch_size, "Number of sampled steps should more than mini batch size."

        # Train for num_sgd_steps iterations (compared to A2C which only
        # train one iteration)
        for e in range(self.ppo_epoch):
            data_generator = rollout.feed_forward_generator(advantages, self.mini_batch_size)

            for sample in data_generator:
                total_loss, policy_loss, value_loss, dist_entropy, ratio = self.compute_loss(sample)
                self.optimizer.zero_grad()
                total_loss.backward()
                if self.config.grad_norm_max:
                    norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_max)
                    norm = norm.item()
                else:
                    norm = 0.0
                self.optimizer.step()

                value_loss_epoch.append(value_loss.item())
                policy_loss_epoch.append(policy_loss.item())
                total_loss_epoch.append(total_loss.item())
                dist_entropy_epoch.append(dist_entropy.item())
                norm_epoch.append(norm)
                ratio_epoch.append(ratio.item())

        return np.mean(policy_loss_epoch), np.mean(value_loss_epoch), np.mean(dist_entropy_epoch), \
               np.mean(total_loss_epoch), np.mean(norm_epoch), adv_mean, np.mean(ratio_epoch)
