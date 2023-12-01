"""
Training script for PPO in the single-agent environment.

This file is almost identical to assignment 3 train_ppo.py. Differences:

1. rename `num_envs` argument to `num_processes` to avoid confusion.
2. remove `env_id` argument to avoid issue in assignment 3 and initialize the environment explicitly here.
3. We move make_envs function in this file so that you can easily change the setting of the environment.

-----
2023 fall quarter, CS260R: Reinforcement Learning.
Department of Computer Science at University of California, Los Angeles.
Course Instructor: Professor Bolei ZHOU.
Assignment Author: Zhenghao PENG.
"""
import argparse
from collections import defaultdict

import gymnasium as gym
import numpy as np
import torch
import tqdm
from metadrive.envs.marl_envs.marl_racing_env import MultiAgentRacingEnv

from agents import load_policies
from core.envs import make_envs
from core.ppo_trainer import PPOConfig
from core.utils import pretty_print, Timer, step_envs

parser = argparse.ArgumentParser()
parser.add_argument(
    "--agent-name",
    default="example_agent",
    type=str,
    help="The name of the agent to be evaluated, aka the subfolder name in 'agents/'. Default: example_agent"
)
parser.add_argument(
    "--log-dir",
    default="data/",
    type=str,
    help="The directory where you want to store the data. "
         "Default: ./data/"
)
parser.add_argument(
    "--num-processes",
    default=10,
    type=int,
    help="The number of parallel RL environments. Default: 10"
)
parser.add_argument(
    "--num-episodes-per-processes",
    default=10,
    type=int,
    help="The number of episode to evaluate per process. Default: 10"
)
parser.add_argument(
    "--seed",
    default=0,
    type=int,
    help="The random seed. Default: 0"
)
parser.add_argument(
    "--render",
    action="store_true",
    help="Whether to launch both the top-down renderer and the 3D renderer. Default: False."
)
args = parser.parse_args()


class SingleAgentRacingEnv(MultiAgentRacingEnv):
    """
    MetaDrive provides a MultiAgentRacingEnv class, where all the input/output data is dict. This wrapper class let the
    environment "behaves like a single-agent RL environment" by unwrapping the output dicts from the environment and
    wrapping the action to be a dict for feeding to the environment.
    """

    AGENT_NAME = "agent0"

    def __init__(self, config):
        assert config["num_agents"] == 1
        super(SingleAgentRacingEnv, self).__init__(config)

    @property
    def action_space(self) -> gym.Space:
        return super(SingleAgentRacingEnv, self).action_space[self.AGENT_NAME]

    @property
    def observation_space(self) -> gym.Space:
        return super(SingleAgentRacingEnv, self).observation_space[self.AGENT_NAME]

    def reset(self, *args, **kwargs):
        obs, info = super(SingleAgentRacingEnv, self).reset(*args, **kwargs)
        return obs[self.AGENT_NAME], info[self.AGENT_NAME]

    def step(self, action):
        o, r, tm, tc, i = super(SingleAgentRacingEnv, self).step({self.AGENT_NAME: action})
        return o[self.AGENT_NAME], r[self.AGENT_NAME], tm[self.AGENT_NAME], tc[self.AGENT_NAME], i[self.AGENT_NAME]

    def reward_function(self, vehicle_id):
        """Only the longitudinal movement is in the reward."""
        vehicle = self.vehicles[vehicle_id]
        step_info = dict()
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
        else:
            current_lane = vehicle.navigation.current_ref_lanes[0]
        longitudinal_last, _ = current_lane.local_coordinates(vehicle.last_position)
        longitudinal_now, lateral_now = current_lane.local_coordinates(vehicle.position)
        self.movement_between_steps[vehicle_id].append(abs(longitudinal_now - longitudinal_last))
        reward = longitudinal_now - longitudinal_last
        step_info["progress"] = (longitudinal_now - longitudinal_last)
        step_info["speed_km_h"] = vehicle.speed_km_h
        step_info["step_reward"] = reward
        step_info["crash_sidewalk"] = False
        if vehicle.crash_sidewalk:
            step_info["crash_sidewalk"] = True
        return reward, step_info

if __name__ == '__main__':
    # Verify algorithm and config

    config = PPOConfig()

    config.num_processes = args.num_processes

    # Seed the environments and setup torch
    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_num_threads(1)

    num_processes = args.num_processes
    render = args.render
    if render:
        assert num_processes == 1

    # Create environments
    def single_env_factory():
        return SingleAgentRacingEnv(dict(
            num_agents=1,
        ))

    envs = make_envs(
        single_env_factory=single_env_factory,
        num_envs=num_processes,
        asynchronous=True,
    )
    total_episodes_to_eval = args.num_episodes_per_processes * num_processes

    # Instantiate all policies here.
    all_policies = load_policies()

    # We will use the specified agent.
    agent_name = args.agent_name
    policy = all_policies[agent_name]()

    print("==================================================")
    print(f"EVALUATING AGENT {agent_name} (CREATOR: {policy.CREATOR_NAME}, UID: {policy.CREATOR_UID})")
    print("==================================================")

    # Setup some stats helpers
    episode_rewards = np.zeros([num_processes, 1], dtype=float)
    total_episodes = total_steps = iteration = 0
    last_total_steps = 0

    result_recorder = defaultdict(list)

    sample_timer = Timer()
    process_timer = Timer()
    update_timer = Timer()
    total_timer = Timer()
    progress = []

    print("Start evaluation!")
    obs = envs.reset()

    if hasattr(policy, "reset"):
        policy.reset()

    with tqdm.tqdm(total=int(total_episodes_to_eval)) as pbar:
        while True:

            cpu_actions = policy(obs)

            # Step the environment
            obs, reward, done, info, masks, total_episodes, \
            total_steps, episode_rewards = step_envs(
                cpu_actions=cpu_actions,
                envs=envs,
                episode_rewards=episode_rewards,
                result_recorder=result_recorder,
                total_steps=total_steps,
                total_episodes=total_episodes,
                device=config.device
            )

            if hasattr(policy, "reset"):
                policy.reset(done_batch=done)

            if render:
                envs.render(mode="topdown")

            pbar.update(total_episodes - pbar.n)
            if total_episodes >= total_episodes_to_eval:
                break

    print("==================================================")
    print(f"THE PERFORMANCE OF {agent_name}:")
    pretty_print({k: np.mean(v) for k, v in result_recorder.items()})
    print("==================================================")

    envs.close()
