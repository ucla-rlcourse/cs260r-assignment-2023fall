"""
This file implements PPO algorithm.

You need to implement `compute_action` and `compute_loss` function.

-----

2023 fall quarter, CS260R: Reinforcement Learning.
Department of Computer Science at University of California, Los Angeles.
Course Instructor: Professor Bolei ZHOU.
Assignment Author: Zhenghao PENG.
"""
import os
import os.path as osp
import sys

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

current_dir = osp.join(osp.abspath(osp.dirname(__file__)))
sys.path.append(current_dir)
sys.path.append(osp.dirname(current_dir))
print(current_dir)

from buffer import PPORolloutStorage
from network import PPOModel


class PPOConfig:
    """Not like previous assignment where we use a dict as config, here we
    build a class to represent config."""

    def __init__(self):
        # Common
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.save_freq = 10
        self.log_freq = 1
        self.num_envs = 1

        # Sample
        self.num_steps = 1000  # num_steps * num_envs = sample_batch_size

        # Learning
        self.gamma = 0.99
        self.lr = 5e-5
        self.grad_norm_max = 10.0
        self.entropy_loss_weight = 0.0
        self.ppo_epoch = 10
        self.mini_batch_size = 256
        self.ppo_clip_param = 0.2
        self.use_gae = True
        self.gae_lambda = 0.95
        self.value_loss_weight = 1.0


class PPOTrainer:
    def __init__(self, env, config):

        self.device = config.device
        self.config = config
        self.lr = config.lr
        self.num_envs = config.num_envs
        self.gamma = config.gamma
        self.num_steps = config.num_steps

        self.grad_norm_max = config.grad_norm_max
        self.value_loss_weight = config.value_loss_weight
        self.entropy_loss_weight = config.entropy_loss_weight

        if isinstance(env.action_space, gym.spaces.Box):
            # Continuous action space
            self.discrete = False
        else:
            self.discrete = True

        if isinstance(env.observation_space, gym.spaces.Tuple):
            num_feats = env.observation_space[0].shape
            self.num_actions = env.action_space[0].n
        else:
            num_feats = env.observation_space.shape
            if self.discrete:
                self.num_actions = env.action_space.n
            else:
                self.num_actions = env.action_space.shape[0]
        self.num_feats = num_feats  # (num channel, width, height)

        self.setup_model_and_optimizer()

        self.act_dim = 1 if self.discrete else self.num_actions

        self.rollouts = PPORolloutStorage(
            self.num_steps, self.num_envs, self.act_dim, self.num_feats[0], self.device, self.discrete,
            self.config.use_gae, self.config.gae_lambda
        )

        # There configs are only used in PPO
        self.ppo_epoch = config.ppo_epoch
        self.mini_batch_size = config.mini_batch_size
        self.clip_param = config.ppo_clip_param

    def setup_model_and_optimizer(self):
        self.model = PPOModel(self.num_feats[0], self.num_actions, self.discrete)
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

        # TODO: Get the actions and the log probability of the action from the output of the neural network (self.model)
        #  Hint: Use proper torch distribution to help you
        actions, action_log_probs = None, None

        if self.discrete:  # Please use categorical distribution.
            logits, values = self.model(obs)
            pass


            actions = actions.view(-1, 1)  # In discrete case only return the chosen action.

        else:  # Please use normal distribution.
            means, log_std, values = self.model(obs)
            pass


            actions = actions.view(-1, self.num_actions)

        values = values.view(-1, 1)
        action_log_probs = action_log_probs.view(-1, 1)

        return values, actions, action_log_probs

    def evaluate_actions(self, obs, act):
        """Run models to get the values, log probability and action
        distribution entropy of the action in current state"""

        obs = self.process_obs(obs)

        if self.discrete:
            assert not torch.is_floating_point(act)
            logits, values = self.model(obs)
            pass
            dist = Categorical(logits=logits)
            action_log_probs = dist.log_prob(act.view(-1)).view(-1, 1)
            dist_entropy = dist.entropy()
        else:
            assert torch.is_floating_point(act)
            means, log_std, values = self.model(obs)
            pass
            action_std = torch.exp(log_std)
            dist = torch.distributions.Normal(means, action_std)
            action_log_probs_raw = dist.log_prob(act)
            action_log_probs = action_log_probs_raw.sum(axis=-1)
            dist_entropy = dist.entropy().sum(-1)

        values = values.view(-1, 1)
        action_log_probs = action_log_probs.view(-1, 1)

        return values, action_log_probs, dist_entropy

    def compute_values(self, obs):
        """Compute the values corresponding to current policy at current
        state"""
        obs = self.process_obs(obs)
        if self.discrete:
            _, values = self.model(obs)
        else:
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
            print("Successfully load weights from {}!".format(save_path))
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

        # TODO: Implement policy loss
        policy_loss = None
        ratio = None  # The importance sampling factor, the ratio of new policy prob over old policy prob
        pass

        policy_loss_mean = policy_loss.mean()

        # [TODO] Implement value loss
        # value_loss = None
        # pass

        value_loss_mean = value_loss.mean()

        # else:
        # value_loss = 0.5 * (return_batch - values).pow(2).mean()

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


if __name__ == '__main__':
    # You can run the script here to see if your PPO is implemented correctly

    from utils import register_metadrive
    from envs import make_envs

    env_name = "MetaDrive-Tut-Easy-v0"

    register_metadrive()


    class FakeConfig(PPOConfig):
        def __init__(self):
            super(FakeConfig, self).__init__()

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.num_envs = 1
            self.num_steps = 200
            self.gamma = 0.99
            self.lr = 5e-4
            self.grad_norm_max = 10.0
            self.value_loss_weight = 1.0
            self.entropy_loss_weight = 0.0


    env = make_envs("CartPole-v0", num_envs=3)
    trainer = PPOTrainer(env, FakeConfig())
    obs = env.reset()
    # Input single observation
    values, actions, action_log_probs = trainer.compute_action(obs[0], deterministic=True)
    new_values, new_action_log_probs, dist_entropy = trainer.evaluate_actions(obs[0], actions)
    assert actions.shape == (1, 1), actions.shape
    assert values.shape == (1, 1), values.shape
    assert action_log_probs.shape == (1, 1), action_log_probs.shape
    assert (values == new_values).all()
    assert (action_log_probs == new_action_log_probs).all()

    # Input multiple observations
    values, actions, action_log_probs = trainer.compute_action(obs, deterministic=False)
    new_values, new_action_log_probs, dist_entropy = trainer.evaluate_actions(obs, actions)
    assert actions.shape == (3, 1), actions.shape
    assert values.shape == (3, 1), values.shape
    assert action_log_probs.shape == (3, 1), action_log_probs.shape
    assert (values == new_values).all()
    assert (action_log_probs == new_action_log_probs).all()

    print("Base trainer discrete case test passed!")
    env.close()

    # ===== Continuous case =====
    env = make_envs("BipedalWalker-v3", asynchronous=False, num_envs=3)
    trainer = PPOTrainer(env, FakeConfig())
    obs = env.reset()
    # Input single observation
    values, actions, action_log_probs = trainer.compute_action(obs[0], deterministic=True)
    new_values, new_action_log_probs, dist_entropy = trainer.evaluate_actions(obs[0], actions)
    assert env.envs[0].action_space.shape[0] == actions.shape[1]
    assert values.shape == (1, 1), values.shape
    assert action_log_probs.shape == (1, 1), action_log_probs.shape
    assert (values == new_values).all()
    assert (action_log_probs == new_action_log_probs).all()

    # Input multiple observations
    values, actions, action_log_probs = trainer.compute_action(obs, deterministic=False)
    new_values, new_action_log_probs, dist_entropy = trainer.evaluate_actions(obs, actions)
    assert env.envs[0].action_space.shape[0] == actions.shape[1]
    assert values.shape == (3, 1), values.shape
    assert action_log_probs.shape == (3, 1), action_log_probs.shape
    assert (values == new_values).all()
    assert (action_log_probs == new_action_log_probs).all()

    print("Base trainer continuous case test passed!")
    env.close()
