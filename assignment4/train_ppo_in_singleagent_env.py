"""
Training script for PPO in the single-agent environment.

This file is almost identical to assignment 3 train_ppo.py. Differences:

1. rename `num_envs` argument to `num_processes` to avoid confusion.
2. remove `env_id` argument to avoid issue in assignment 3 and initialize the environment explicitly here.
3. We move make_envs function in this file so that you can easily change the setting of the environment.

In this script, you are free to adjust anything, such as the environment config, the reward function etc.

-----
2023 fall quarter, CS260R: Reinforcement Learning.
Department of Computer Science at University of California, Los Angeles.
Course Instructor: Professor Bolei ZHOU.
Assignment Author: Zhenghao Mark PENG, Yuxin Liu.
"""
import argparse
import os
from collections import deque

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import tqdm
from metadrive.envs.marl_envs.marl_racing_env import MultiAgentRacingEnv

from core.envs import make_envs
from core.ppo_trainer import PPOTrainer, PPOConfig
from core.utils import verify_log_dir, pretty_print, Timer, step_envs

# You can adjust the environment config here.
# You can check out the environment below to see how the config affects the reward function.
ENVIRONMENT_CONFIG = dict(
    num_agents=1,  # Don't change

    # Reward function
    crash_sidewalk_penalty=50,
    success_reward=100,
    speed_reward=0,
)

parser = argparse.ArgumentParser()
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
    "--seed",
    default=0,
    type=int,
    help="The random seed. Default: 0"
)
parser.add_argument(
    "--max-steps",
    "-N",
    default=1e7,
    type=float,
    help="The random seed. Default: 1e7"
)
parser.add_argument(
    "--num-steps",
    default=2000,
    type=int,
    help="Number of steps to sample at each process. Default: 2000"
)
parser.add_argument(
    "--synchronous",
    action="store_true",
    help="If True, run multiple environments in the main process. Otherwise will fork multiple subprocesses."
)
parser.add_argument(
    "--lr",
    default=-1,
    type=float,
    help="Learning rate. Default: 5e-5"
)
parser.add_argument(
    "--gae-lambda",
    default=0.95,
    type=float,
    help="GAE coefficient. Default: 0.95"
)
parser.add_argument(
    "--num-epoch",
    default=10,
    type=int,
    help="Number of epochs for training PPO in each training iteration. Default: 10"
)
parser.add_argument(
    "--pretrained-model-log-dir",
    default="",
    type=str,
    help="The folder that hosts the pretrained model. Example: agents/youragentname"
)
parser.add_argument(
    "--pretrained-model-suffix",
    default="",
    type=str,
    help="The suffix of the checkpoint (if you are using PPO during pretraining). Example: iter275"
)
args = parser.parse_args()

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(obs):
    obs = torch.from_numpy(obs.astype(np.float32)).to(default_device)
    return obs


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
        """
        Reward function copied from metadrive.envs.marl_envs.mark_racing_env
        You can freely adjust the config or add terms.
        """
        vehicle = self.vehicles[vehicle_id]
        step_info = dict()

        # Reward for moving forward in current lane
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
        else:
            current_lane = vehicle.navigation.current_ref_lanes[0]
        longitudinal_last, _ = current_lane.local_coordinates(vehicle.last_position)
        longitudinal_now, lateral_now = current_lane.local_coordinates(vehicle.position)

        self.movement_between_steps[vehicle_id].append(abs(longitudinal_now - longitudinal_last))

        reward = 0.0
        reward += self.config["driving_reward"] * (longitudinal_now - longitudinal_last)
        reward += self.config["speed_reward"] * (vehicle.speed_km_h / vehicle.max_speed_km_h)

        step_info["progress"] = (longitudinal_now - longitudinal_last)
        step_info["speed_km_h"] = vehicle.speed_km_h

        step_info["step_reward"] = reward
        step_info["crash_sidewalk"] = False
        if self._is_arrive_destination(vehicle):
            reward = +self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward = -self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
        elif vehicle.crash_sidewalk:
            reward = -self.config["crash_sidewalk_penalty"]
            step_info["crash_sidewalk"] = True
        elif self._is_idle(vehicle_id):
            reward = -self.config["idle_penalty"]

        return reward, step_info


if __name__ == '__main__':
    # Verify algorithm and config

    config = PPOConfig()

    config.num_processes = args.num_processes
    config.num_steps = args.num_steps
    config.gae_lambda = args.gae_lambda
    config.ppo_epoch = args.num_epoch
    if args.lr != -1:
        config.lr = args.lr

    # Seed the environments and setup torch
    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_num_threads(1)

    # Clean log directory
    algo = "ppo"
    log_dir = verify_log_dir(args.log_dir, algo)

    # Create vectorized environments
    num_processes = args.num_processes


    def single_env_factory():
        return SingleAgentRacingEnv(ENVIRONMENT_CONFIG)


    envs = make_envs(
        single_env_factory=single_env_factory,
        num_envs=num_processes,
        asynchronous=not args.synchronous,
    )

    # Setup trainer
    trainer = PPOTrainer(config)

    if args.pretrained_model_log_dir:
        assert args.pretrained_model_suffix, "You should also specify --pretrained-model-suffix"
        trainer.load_w(log_dir=args.pretrained_model_log_dir, suffix=args.pretrained_model_suffix)
        print(f"Successfully loaded pretrained model at {args.pretrained_model_log_dir} "
              f"with suffix {args.pretrained_model_suffix}.")
    else:
        print("No pretrained model is loaded. You will train the agent from scratch.")

    # Setup some stats helpers
    episode_rewards = np.zeros([num_processes, 1], dtype=float)
    total_episodes = total_steps = iteration = 0
    last_total_steps = 0

    result_recorder = dict(
        crash_sidewalk_rate=deque(maxlen=30_000),
        crash_vehicle_rate=deque(maxlen=30_000),
        idle_rate=deque(maxlen=30_000),
        speed_km_h=deque(maxlen=30_000),
        max_step_rate=deque(maxlen=100),
        success_rate=deque(maxlen=100),
        episode_reward=deque(maxlen=100),
        episode_length=deque(maxlen=100)
    )

    sample_timer = Timer()
    process_timer = Timer()
    update_timer = Timer()
    total_timer = Timer()
    progress = []

    # Start training
    print("Start training!")
    obs = envs.reset()
    trainer.rollouts.observations[0].copy_(to_tensor(obs))
    with tqdm.tqdm(total=int(args.max_steps)) as pbar:
        while True:
            # ===== Sample Data =====
            with sample_timer:
                for index in range(config.num_steps):
                    with torch.no_grad():
                        values, actions, action_log_prob = \
                            trainer.compute_action(trainer.rollouts.observations[index])

                    assert values.shape == (num_processes, 1)
                    assert action_log_prob.shape == (num_processes, 1)

                    # `actions` is a torch tensor, so we need to turn it into numpy array.
                    cpu_actions = actions.cpu().numpy()

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

                    rewards = torch.from_numpy(
                        reward.astype(np.float32)).view(-1, 1).to(config.device)

                    # Store samples
                    trainer.rollouts.insert(to_tensor(obs), actions, action_log_prob, values, rewards, masks)

            # ===== Process Samples =====
            with process_timer:
                with torch.no_grad():
                    next_value = trainer.compute_values(trainer.rollouts.observations[-1])
                trainer.rollouts.compute_returns(next_value, config.gamma)

            # ===== Update Policy =====
            with update_timer:
                policy_loss, value_loss, dist_entropy, total_loss, norm, adv, ratio = trainer.update(trainer.rollouts)
                trainer.rollouts.after_update()

            # ===== Log information =====
            if iteration % config.log_freq == 0:
                stats = dict(
                    log_dir=log_dir,
                    frame_per_second=int(total_steps / total_timer.now),
                    # episode_reward=np.mean(reward_recorder),
                    # evaluate_stats=evaluate_stat,

                    policy_loss=policy_loss,
                    entropy=dist_entropy,
                    value_loss=value_loss,
                    total_loss=total_loss,
                    grad_norm=norm,
                    adv_mean=adv,
                    ratio=ratio,

                    total_steps=total_steps,
                    total_episodes=total_episodes,
                    iteration=iteration,
                    total_time=total_timer.now
                )

                for k, recorder in result_recorder.items():
                    stats[k] = np.mean(recorder)

                progress.append(stats)
                pretty_print({
                    "===== {} Training Iteration {} =====".format(
                        algo, iteration): stats
                })

            if iteration % config.save_freq == 0:
                trainer_path = trainer.save_w(log_dir, "iter{}".format(iteration))
                progress_path = os.path.join(log_dir, "progress.csv")
                pd.DataFrame(progress).to_csv(progress_path)
                print("Trainer is saved at <{}>. Progress is saved at <{}>.".format(
                    trainer_path, progress_path
                ))

            pbar.update(total_steps - last_total_steps)
            last_total_steps = total_steps
            if total_steps > int(args.max_steps):
                break

            iteration += 1

    trainer.save_w(log_dir, "final")

    envs.close()
