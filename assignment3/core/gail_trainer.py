"""
This file implements GAIL algorithm.

GAIL algorithm heavily relies on PPO algorithm, so we will reuse large
body of code in PPOTrainer.

Major changes:

1. Use GAIL model so the discriminator takes state and action as input
2. Discard values, advantages and value network
3. Implement training pipelines for the discriminator
4. Update the generator (policy network) according to the prediction of discriminator

-----

2023 fall quarter, CS260R: Reinforcement Learning.
Department of Computer Science at University of California, Los Angeles.
Course Instructor: Professor Bolei ZHOU.
Assignment Author: Zhenghao PENG.
"""
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

current_dir = osp.join(osp.abspath(osp.dirname(__file__)))
sys.path.append(current_dir)
sys.path.append(osp.dirname(current_dir))
print(current_dir)

from ppo_trainer import PPOTrainer, PPOConfig
from network import GAILModel
from buffer import ExpertDataset

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(obs):
    obs = torch.from_numpy(obs.astype(np.float32)).to(default_device)
    return obs


class GAILConfig(PPOConfig):
    def __init__(self):
        super(GAILConfig, self).__init__()

        # Hyper-parameters for GAIL
        self.generator_epoch = 10  # << This is original "ppo_epoch" but we rename it for clarity.
        self.generator_lr = 1e-4
        self.discriminator_epoch = 5
        self.discriminator_lr = 1e-4
        self.discriminator_mini_batch_size = 128  # So we have 128 "positive samples" and 128 "negative samples"


class GAILTrainer(PPOTrainer):
    def __init__(self, env, config):
        assert isinstance(config, GAILConfig)
        super(GAILTrainer, self).__init__(env, config)
        assert not self.discrete, "We only implement continuous GAIL only."
        self.expert_dataset = None
        self.expert_dataloader = None

    def setup_model_and_optimizer(self):
        self.model = GAILModel(
            input_size=self.num_feats[0],
            act_dim=self.num_actions,
            output_size=self.num_actions,
            discrete=self.discrete
        )
        self.model = self.model.to(self.device)
        self.model.train()
        self.optimizer = optim.Adam(self.model.get_generator_parameters(), lr=self.config.generator_lr)
        self.optimizer_discriminator = optim.Adam(
            self.model.get_discriminator_parameters(), lr=self.config.discriminator_lr
        )

    def generate_expert_data(self, envs, size=10000):
        """This function generate expert data and prepare self.expert_dataloader."""
        expert_dataset = ExpertDataset(
            size, self.num_envs, self.act_dim, self.num_feats[0], self.device, self.discrete,
        )

        expert = PPOTrainer(envs, PPOConfig())
        expert.load_w(current_dir, suffix="expert_MetaDrive-Tut-Easy-v0")

        num_envs = envs.num_envs
        required_steps = int(size / num_envs)

        episode_rewards = np.ones(num_envs)
        rewards = []
        successes = []

        obs = envs.reset()
        expert_dataset.observations[0].copy_(to_tensor(obs))

        for step in range(1, required_steps + 1):
            values, actions, log_probs = expert.compute_action(obs)

            obs, reward, done, info = envs.step(actions.cpu().numpy())

            episode_rewards += reward.reshape(episode_rewards.shape)
            for idx, d in enumerate(done):
                if d:  # the episode is done
                    # Record the reward of the terminated episode to
                    rewards.append(episode_rewards[idx].copy())
                    successes.append(info[idx].get("arrive_dest", 0))
            masks = 1. - done.astype(np.float32)
            episode_rewards *= masks.reshape(-1, )

            # Build expert dataset
            expert_dataset.insert(
                current_obs=to_tensor(obs),
                action=actions,
                action_log_prob=None,
                value_pred=None,
                reward=None,
                mask=None
            )

            if step % 1000 == 0:
                print(
                    "Generated {}/{} steps from the expert. Average episode reward: {:.3f}, success rate: {:.3f}".format(
                        step, size, np.mean(rewards), np.mean(successes)
                    ))

        self.expert_dataset = expert_dataset
        expert_bs = self.config.discriminator_mini_batch_size
        self.expert_dataloader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=expert_bs,
            shuffle=True,
            drop_last=len(expert_dataset) > expert_bs
        )

        print(
            "Generated {} steps from the expert. Average episode reward: {:.3f}, success rate: {:.3f}".format(
                size, np.mean(rewards), np.mean(successes)
            ))

        del expert

    def compute_action(self, obs, deterministic=False):
        """In GAIL model, we don't have value network."""
        # TODO: Copy and paste major body of the compute_action function from the PPO trainer.

        obs = self.process_obs(obs)

        # TODO: Get the actions and the log probability of the action from the output of neural network.
        #  Hint: Use proper torch distribution to help you
        actions, action_log_probs = None, None

        if self.discrete:  # Please use categorical distribution.
            logits = self.model(obs)
            pass  # TODO

            actions = actions.view(-1, 1)  # In discrete case only return the chosen action.

        else:  # Please use normal distribution.
            means, log_std = self.model(obs)
            pass  # TODO

            actions = actions.view(-1, self.num_actions)

        # values = values.view(-1, 1)
        action_log_probs = action_log_probs.view(-1, 1)

        # return values, actions, action_log_probs
        return actions, action_log_probs

    def evaluate_actions(self, obs, act):
        """Run models to get the values, log probability and action
        distribution entropy of the action in current state"""

        obs = self.process_obs(obs)

        if self.discrete:
            assert not torch.is_floating_point(act)
            logits = self.model(obs)
            pass
            dist = Categorical(logits=logits)
            action_log_probs = dist.log_prob(act.view(-1)).view(-1, 1)
            dist_entropy = dist.entropy()
        else:
            assert torch.is_floating_point(act)
            means, log_std = self.model(obs)
            pass
            action_std = torch.exp(log_std)
            dist = torch.distributions.Normal(means, action_std)
            action_log_probs_raw = dist.log_prob(act)
            action_log_probs = action_log_probs_raw.sum(axis=-1)
            dist_entropy = dist.entropy().sum(-1)

        # values = values.view(-1, 1)
        action_log_probs = action_log_probs.view(-1, 1)

        gail_rewards = self.model.compute_prediction(obs=obs, act=act).detach()

        return gail_rewards, action_log_probs, dist_entropy

    def compute_loss(self, sample):
        """Compute the loss of PPO"""
        observations_batch, actions_batch, value_preds_batch, return_batch, masks_batch, \
            old_action_log_probs_batch, _ = sample

        assert old_action_log_probs_batch.shape == (self.mini_batch_size, 1)
        # assert adv_targ.shape == (self.mini_batch_size, 1)
        assert return_batch.shape == (self.mini_batch_size, 1)

        # Here we computed gail_rewards:
        gail_rewards, action_log_probs, dist_entropy = self.evaluate_actions(observations_batch, actions_batch)

        assert gail_rewards.shape == (self.mini_batch_size, 1)
        assert action_log_probs.shape == (self.mini_batch_size, 1)
        assert not gail_rewards.requires_grad, "gail_rewards should has no gradient when updating policy!"
        assert action_log_probs.requires_grad
        assert dist_entropy.requires_grad

        # TODO: Implement policy loss
        # Hint: Copy the relevant code from ppo_trainer.py but treat gail_rewards as the advantage.
        # No need to implement value network, value loss and advantage as we don't use value estimation for GAIL.
        policy_loss = None
        ratio = None  # The importance sampling factor, the ratio of new policy prob over old policy prob
        pass

        policy_loss_mean = policy_loss.mean()

        # This is the total loss
        loss = policy_loss - self.config.entropy_loss_weight * dist_entropy
        loss = loss.mean()

        return loss, policy_loss_mean, torch.mean(gail_rewards), torch.mean(dist_entropy), torch.mean(ratio)

    def update(self, rollout):
        # Before PPO training starts, train discriminator and value network first.

        assert self.expert_dataloader is not None, "Please call trainer.generate_expert_data before training!"

        # train discriminator
        d_loss_list = []

        discriminator_loss_func = nn.BCELoss()

        for epoch in range(1, self.config.discriminator_epoch + 1):
            data_generator = rollout.feed_forward_generator(None, self.config.discriminator_mini_batch_size)
            for (expert_data, agent_data) in zip(self.expert_dataloader, data_generator):
                # In each epoch, split the data collected from environment into several minibatch
                # Find an equally-sized batch from expert dataset
                # And train the discriminator to label these two sets of data
                agent_generated_obs = agent_data[0]
                agent_generated_actions = agent_data[1]
                expert_generated_obs = expert_data[0]
                expert_generated_actions = expert_data[1]

                # TODO: Call the discriminator to get the prediction of agent and expert's state-action pairs.
                #  and flatten the tensor by calling .flatten()
                agent_prediction = None
                expert_prediction = None
                pass

                assert agent_prediction.dim() == 1
                assert expert_prediction.dim() == 1

                # TODO: Compute the discriminator loss using discriminator_loss_func.
                # Hint: We should assume the ground-truth label for all agent_prediction to be 0 and
                # expert_prediction to be 1. This is the essence of GAIL.
                discriminator_loss = None
                pass

                # For stats
                with torch.no_grad():
                    d_loss_list.append(discriminator_loss.item())

                # Update discriminator
                self.optimizer_discriminator.zero_grad()
                discriminator_loss.backward()
                self.optimizer_discriminator.step()

        discriminator_loss_mean = sum(d_loss_list) / len(d_loss_list)

        # === Training the generator (policy network) ===
        gail_reward_mean_epoch = []
        policy_loss_epoch = []
        dist_entropy_epoch = []
        total_loss_epoch = []
        norm_epoch = []
        ratio_epoch = []
        for e in range(self.config.generator_epoch):
            data_generator = rollout.feed_forward_generator(None, self.mini_batch_size)
            for sample in data_generator:
                total_loss, policy_loss, gail_reward_mean, dist_entropy, ratio = self.compute_loss(sample)
                self.optimizer.zero_grad()
                total_loss.backward()
                if self.config.grad_norm_max:
                    norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_max)
                    norm = norm.item()
                else:
                    norm = 0.0
                self.optimizer.step()

                gail_reward_mean_epoch.append(gail_reward_mean.item())
                policy_loss_epoch.append(policy_loss.item())
                total_loss_epoch.append(total_loss.item())
                dist_entropy_epoch.append(dist_entropy.item())
                norm_epoch.append(norm)
                ratio_epoch.append(ratio.item())

        return np.mean(policy_loss_epoch), discriminator_loss_mean, np.mean(gail_reward_mean_epoch), \
            np.mean(dist_entropy_epoch), np.mean(total_loss_epoch), np.mean(norm_epoch), \
            None, np.mean(ratio_epoch)


if __name__ == '__main__':
    env_name = "MetaDrive-Tut-Hard-v0"
    from utils import register_metadrive
    from envs import make_envs

    register_metadrive()


    class FakeConfig(GAILConfig):
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
    trainer = GAILTrainer(env, FakeConfig())
    obs, _ = env.reset()
    # Input single observation
    values, actions, action_log_probs = trainer.compute_action(obs[0], deterministic=True)
    new_values, new_action_log_probs, dist_entropy = trainer.evaluate_actions(obs[0], actions)
    assert actions.shape == (1, 1), actions.shape
    assert action_log_probs.shape == (1, 1), action_log_probs.shape
    assert (action_log_probs == new_action_log_probs).all()

    # Input multiple observations
    values, actions, action_log_probs = trainer.compute_action(obs, deterministic=False)
    new_values, new_action_log_probs, dist_entropy = trainer.evaluate_actions(obs, actions)
    assert actions.shape == (3, 1), actions.shape
    assert action_log_probs.shape == (3, 1), action_log_probs.shape
    assert (action_log_probs == new_action_log_probs).all()

    print("Base trainer discrete case test passed!")
    env.close()

    # ===== Continuous case =====
    env = make_envs("BipedalWalker-v3", asynchronous=False, num_envs=3)
    trainer = GAILTrainer(env, FakeConfig())
    obs, _ = env.reset()
    # Input single observation
    values, actions, action_log_probs = trainer.compute_action(obs[0], deterministic=True)
    new_values, new_action_log_probs, dist_entropy = trainer.evaluate_actions(obs[0], actions)
    assert env.envs[0].action_space.shape[0] == actions.shape[1]
    assert action_log_probs.shape == (1, 1), action_log_probs.shape
    assert (action_log_probs == new_action_log_probs).all()

    # Input multiple observations
    values, actions, action_log_probs = trainer.compute_action(obs, deterministic=False)
    new_values, new_action_log_probs, dist_entropy = trainer.evaluate_actions(obs, actions)
    assert env.envs[0].action_space.shape[0] == actions.shape[1]
    assert action_log_probs.shape == (3, 1), action_log_probs.shape
    assert (action_log_probs == new_action_log_probs).all()

    print("Base trainer continuous case test passed!")
    env.close()
