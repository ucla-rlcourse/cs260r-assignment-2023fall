import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class BaseRolloutStorage:
    def __init__(self, num_steps, num_processes, act_dim, obs_dim, device, discrete):
        def zeros(*shapes):
            return torch.zeros(*shapes).to(torch.float32).to(device)

        self.observations = zeros(num_steps + 1, num_processes, obs_dim)
        self.rewards = zeros(num_steps, num_processes, 1)
        self.value_preds = zeros(num_steps + 1, num_processes, 1)
        self.returns = zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = zeros(num_steps, num_processes, 1)
        if discrete:
            self.actions = zeros(num_steps, num_processes, 1)
            self.actions = self.actions.to(torch.long)
        else:
            self.actions = zeros(num_steps, num_processes, act_dim)
        self.masks = torch.ones(num_steps + 1, num_processes, 1).to(device)

        self.num_steps = num_steps
        self.step = 0

    def insert(self, current_obs, action, action_log_prob, value_pred, reward, mask):
        self.observations[self.step + 1].copy_(current_obs)
        self.actions[self.step].copy_(action)

        # To fit the GAIL case
        if action_log_prob is not None:
            self.action_log_probs[self.step].copy_(action_log_prob)
        if value_pred is not None:
            self.value_preds[self.step].copy_(value_pred)
        if reward is not None:
            self.rewards[self.step].copy_(reward)
        if mask is not None:
            self.masks[self.step + 1].copy_(mask)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])


class PPORolloutStorage(BaseRolloutStorage):
    def __init__(self, num_steps, num_processes, act_dim, obs_dim, device, discrete,
                 use_gae=True, gae_lambda=0.95):
        super().__init__(num_steps, num_processes, act_dim, obs_dim, device, discrete=discrete)
        self.gae = use_gae
        self.gae_lambda = gae_lambda

    def feed_forward_generator(self, advantages, mini_batch_size):
        """A generator to provide samples for PPO. PPO run SGD for multiple
        times so we need more efforts to prepare data for it."""
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                               mini_batch_size, drop_last=True)
        for indices in sampler:
            observations_batch = self.observations[:-1].view(-1, *self.observations.size()[2:])[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            if advantages is not None:
                adv_targ = advantages.view(-1, 1)[indices]
            else:
                adv_targ = None
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]

            yield observations_batch, actions_batch, value_preds_batch, return_batch, \
                  masks_batch, old_action_log_probs_batch, adv_targ

    def compute_returns(self, next_value, gamma):
        if self.gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                # TODO: Implement GAE advantage computing here.
                # Hint:
                # How to compute the self.returns[t]?
                #  * The return at timestep t should be (advantage_t + value_t)
                # How to compute advantage_t?
                #  * To compute advantage_t, we use the variable `gae`. Starting from t = T, we iterate over
                #    t=T, t=T-1, t=T-2, ... and keep updating `gae` from back to forth.
                #  * The `gae` contains two terms, one is current step's TD error, another is the bootstrapping `gae`.
                #  * You should use reward_t, value_t, mask_{t+1} to compute the TD error at this step.
                #  * The TD error is basically reward + gamma * next value * next mask - current value.
                #  * Then, we need to combine the TD error with `advantage_{t+1}` (which is the `gae` itself as it's
                #    computed for t+1) to get the new `gae` as the `advantage_t`.
                # self.returns[step] = ???
                pass
                delta = (
                        self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                        - self.value_preds[step]
                )
                gae = delta + gamma * self.gae_lambda * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            raise NotImplementedError()


class ExpertDataset(BaseRolloutStorage, torch.utils.data.Dataset):
    def __len__(self):
        return self.num_steps

    def __getitem__(self, i):
        return self.observations[i], self.actions[i]
