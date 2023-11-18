"""
TD3 Trainer.

-----
2023 fall quarter, CS260R: Reinforcement Learning.
Department of Computer Science at University of California, Los Angeles.
Course Instructor: Professor Bolei ZHOU.
Assignment Author: Zhenghao PENG.
"""
import copy
import os
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

current_dir = osp.join(osp.abspath(osp.dirname(__file__)))
sys.path.append(current_dir)
sys.path.append(osp.dirname(current_dir))
print(current_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


class TD3Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(TD3Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class TD3Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(TD3Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3Trainer:
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            lr=5e-5  # Small LR in TD3 is important to train in MetaDrive!
    ):
        self.actor = TD3Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = TD3Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def select_action_in_batch(self, state):
        state = torch.FloatTensor(state).to(device)
        return self.actor(state).cpu().data.numpy()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Following the TODOs below to implement critic loss
        with torch.no_grad():
            # TODO: Generate noise and clipped the noise.
            # Hint: Sample a noise from Normal distribution with the scale self.policy_noise.
            # noise = (
            #         ???
            # ).clamp(-self.noise_clip, self.noise_clip)
            pass

            # TODO: Select next action according to the delayed-updated policy (self.actor_target) and add noise.
            # next_action = (
            #         ???
            # ).clamp(-self.max_action, self.max_action)
            pass

            # TODO: Compute the target Q value (the objective of both critics).
            # Hint: Call the delayed-updated critic (self.critic_target) first, then compute the critic objective.
            target_Q = None
            pass

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # TODO: Compute critic loss.
        # Hint: Compute the MSE for both critics and add them up.
        critic_loss = None
        pass

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = None

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # TODO: Compute actor loss
            actor_loss = None
            pass


            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "actor_loss": np.nan if actor_loss is None else actor_loss.item(),
            "critic_loss": np.nan if critic_loss is None else critic_loss.item(),
            "target_q": target_Q.mean().item(),
            "current_q1": current_Q1.mean().item(),
            "current_q2": current_Q2.mean().item(),
            "reward": reward.mean().item(),
        }

    def save(self, filename):
        torch.save(self.critic.state_dict(), os.path.join(filename, "critic"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(filename, "critic_optimizer"))

        torch.save(self.actor.state_dict(), os.path.join(filename, "actor"))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(filename, "actor_optimizer"))

    def load(self, filename):
        self.critic.load_state_dict(torch.load(os.path.join(filename, "critic")))
        self.critic_optimizer.load_state_dict(torch.load(os.path.join(filename, "critic_optimizer")))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(os.path.join(filename, "actor")))
        self.actor_optimizer.load_state_dict(torch.load(os.path.join(filename, "actor_optimizer")))
        self.actor_target = copy.deepcopy(self.actor)
