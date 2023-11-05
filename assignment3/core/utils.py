"""
This file implements some helper functions.

-----

2023 fall quarter, CS260R: Reinforcement Learning.
Department of Computer Science at University of California, Los Angeles.
Course Instructor: Professor Bolei ZHOU.
Assignment Author: Zhenghao PENG.
"""
import copy
import glob
import json
import os
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import yaml


def verify_log_dir(log_dir, *others):
    if others:
        log_dir = os.path.join(log_dir, *others)
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)
    return os.path.abspath(log_dir)


def step_envs(cpu_actions, envs, episode_rewards, reward_recorder, success_recorder, total_steps, total_episodes,
              device):
    """Step the vectorized environments for one step. Process the reward
    recording and terminal states."""
    obs, reward, done, info = envs.step(cpu_actions)
    episode_rewards += reward.reshape(episode_rewards.shape)
    episode_rewards_old_shape = episode_rewards.shape
    if not np.isscalar(done[0]):
        done = np.all(done, axis=1)
    for idx, d in enumerate(done):
        if d:  # the episode is done
            # Record the reward of the terminated episode to
            reward_recorder.append(episode_rewards[idx].copy())
            if "arrive_dest" in info[idx]:
                success_recorder.append(info[idx]["arrive_dest"])
            total_episodes += 1
    masks = 1. - done.astype(np.float32)
    episode_rewards *= masks.reshape(-1, 1)
    assert episode_rewards.shape == episode_rewards_old_shape
    total_steps += obs[0].shape[0] if isinstance(obs, tuple) else obs.shape[0]
    masks = torch.from_numpy(masks).to(device).view(-1, 1)
    return obs, reward, done, info, masks, total_episodes, total_steps, episode_rewards


def flatten_dict(dt, delimiter="/"):
    dt = copy.deepcopy(dt)
    while any(isinstance(v, dict) for v in dt.values()):
        remove = []
        add = {}
        for key, value in dt.items():
            if isinstance(value, dict):
                for subkey, v in value.items():
                    add[delimiter.join([key, subkey])] = v
                remove.append(key)
        dt.update(add)
        for k in remove:
            del dt[k]
    return dt


def evaluate(trainer, eval_envs, num_episodes=10, seed=0):
    """This function evaluate the given policy and return the mean episode
    reward.
    :param policy: a function whose input is the observation
    :param env: an environment instance
    :param num_episodes: number of episodes you wish to run
    :param seed: the random seed
    :return: the averaged episode reward of the given policy.
    """

    def get_action(obs):
        with torch.no_grad():
            act = trainer.compute_action(obs, deterministic=True)[1]
        if trainer.discrete:
            act = act.view(-1).cpu().numpy()
        else:
            act = act.cpu().numpy()
        return act

    reward_recorder = []
    episode_length_recorder = []
    episode_rewards = np.zeros([eval_envs.num_envs, 1], dtype=float)
    total_steps = 0
    total_episodes = 0
    eval_envs.seed(seed)
    obs = eval_envs.reset()
    while True:
        obs, reward, done, info, masks, total_episodes, total_steps, \
            episode_rewards = step_envs(
            get_action(obs), eval_envs, episode_rewards, reward_recorder, episode_length_recorder,
            total_steps, total_episodes, trainer.device)
        if total_episodes >= num_episodes:
            break
    return reward_recorder, episode_length_recorder


class Timer:
    def __init__(self, interval=10):
        self.value = 0.0
        self.start = time.time()
        self.buffer = deque(maxlen=interval)

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.value = time.time() - self.start
        self.buffer.append(self.value)

    @property
    def now(self):
        """Return the seconds elapsed since initializing this class"""
        return time.time() - self.start

    @property
    def avg(self):
        return np.mean(self.buffer, dtype=float)


def pretty_print(result):
    result = result.copy()
    out = {}
    for k, v in result.items():
        if v is not None:
            out[k] = v
    cleaned = json.dumps(out)
    print("\n", yaml.safe_dump(json.loads(cleaned), default_flow_style=False))


def register_metadrive():
    try:
        from metadrive.envs import MetaDriveEnv
        from metadrive.utils.config import merge_config_with_unknown_keys
    except ImportError as e:
        print("Please install MetaDrive through: pip install git+https://github.com/decisionforce/metadrive")
        raise e

    env_names = []
    try:
        class MetaDriveEnvTut(gym.Wrapper):
            def __init__(self, config, *args, render_mode=None, **kwargs):
                # Ignore render_mode
                self._render_mode = render_mode
                super().__init__(MetaDriveEnv(config))

                if isinstance(self.env.action_space, gym.spaces.Discrete):
                    self.action_space = gym.spaces.Discrete(int(np.prod(self.env.action_space.n)))
                else:
                    self.action_space = self.env.action_space

            def reset(self, *args, seed=None, render_mode=None, options=None, **kwargs):
                # Ignore seed and render_mode
                return self.env.reset(*args, **kwargs)

            def render(self):
                return self.env.render(mode=self._render_mode)

        def _make_env(*args, **kwargs):
            return MetaDriveEnvTut(*args, **kwargs)

        env_name = "MetaDrive-Tut-Easy-v0"
        gym.register(id=env_name, entry_point=_make_env, kwargs={"config": dict(
            map="S",
            environment_num=1,
            horizon=200,
            start_seed=1000,
        )})
        env_names.append(env_name)

        env_name = "MetaDrive-Tut-Hard-v0"
        gym.register(id=env_name, entry_point=_make_env, kwargs={"config": dict(
            start_seed=1000,
            environment_num=20,
            horizon=1000,
        )})
        env_names.append(env_name)

        for env_num in [1, 5, 10, 20, 50, 100]:
            env_name = "MetaDrive-Tut-{}Env-v0".format(env_num)
            gym.register(id=env_name, entry_point=_make_env, kwargs={"config": dict(
                start_seed=0,
                environment_num=env_num,
                horizon=1000,
            )})
            env_names.append(env_name)

        env_name = "MetaDrive-Tut-Test-v0".format(env_num)
        gym.register(id=env_name, entry_point=_make_env, kwargs={"config": dict(
            start_seed=1000,
            environment_num=50,
            horizon=1000,
        )})
        env_names.append(env_name)

    except gym.error.Error as e:
        print("Information when registering MetaDrive: ", e)
    else:
        print("Successfully registered MetaDrive environments: ", env_names)


if __name__ == '__main__':
    # Test
    register_metadrive()
    env = gym.make("MetaDrive-Tut-Easy-v0", config={'use_render': True})
    env.reset()
