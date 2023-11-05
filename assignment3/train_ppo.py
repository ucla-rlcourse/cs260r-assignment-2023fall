"""
Training script for PPO

-----
2023 fall quarter, CS260R: Reinforcement Learning.
Department of Computer Science at University of California, Los Angeles.
Course Instructor: Professor Bolei ZHOU.
Assignment Author: Zhenghao PENG.
"""
import argparse
import os
from collections import deque

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import tqdm

from core.envs import make_envs
from core.ppo_trainer import PPOTrainer, PPOConfig
from core.utils import verify_log_dir, pretty_print, Timer, step_envs

gym.logger.set_level(40)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--log-dir",
    default="data/",
    type=str,
    help="The directory where you want to store the data. "
         "Default: ./data/"
)
parser.add_argument(
    "--num-envs",
    default=10,
    type=int,
    help="The number of parallel environments. Default: 10"
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
)
parser.add_argument(
    "--env-id",
    default="MetaDrive-Tut-Easy-v0",
    type=str,
)
parser.add_argument(
    "--synchronous",
    action="store_true"
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
    "--restore",
    action="store_true",
    help="Whether to load checkpoint-final and continue learning. Default: False"
)
args = parser.parse_args()

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(obs):
    obs = torch.from_numpy(obs.astype(np.float32)).to(default_device)
    return obs


if __name__ == '__main__':
    # Verify algorithm and config

    config = PPOConfig()

    config.num_envs = args.num_envs
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
    num_envs = args.num_envs
    env_id = args.env_id
    envs = make_envs(
        env_id=env_id,
        log_dir=log_dir,
        num_envs=num_envs,
        asynchronous=not args.synchronous,
    )

    # Setup trainer
    trainer = PPOTrainer(envs, config)

    if args.restore:
        # Try to reload models
        trainer.load_w(log_dir, "final")

    # Setup some stats helpers
    episode_rewards = np.zeros([num_envs, 1], dtype=float)
    total_episodes = total_steps = iteration = 0
    last_total_steps = 0

    reward_recorder = deque(maxlen=100)
    success_recorder = deque(maxlen=100)
    sample_timer = Timer()
    process_timer = Timer()
    update_timer = Timer()
    total_timer = Timer()
    progress = []
    # evaluate_stat = {}

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

                    assert values.shape == (num_envs, 1)
                    assert action_log_prob.shape == (num_envs, 1)

                    # `actions` is a torch tensor, so we need to turn it into numpy array.
                    cpu_actions = actions.cpu().numpy()
                    if trainer.discrete:
                        cpu_actions = cpu_actions.reshape(-1)  # flatten

                    # Step the environment
                    # (Check step_envs function, you need to implement it)
                    obs, reward, done, info, masks, total_episodes, \
                        total_steps, episode_rewards = step_envs(
                        cpu_actions, envs, episode_rewards,
                        reward_recorder, success_recorder, total_steps,
                        total_episodes, config.device
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
                    episode_reward=np.mean(reward_recorder),
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

                if success_recorder:
                    stats["success_rate"] = np.mean(success_recorder)

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
