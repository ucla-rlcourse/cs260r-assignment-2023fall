import argparse
import os
from collections import deque

import numpy as np
import pandas as pd
import torch
import tqdm

from core.envs import make_envs
from core.td3_trainer import ReplayBuffer, TD3Trainer
from core.utils import pretty_print, Timer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hyper-parameters that we need to worry about
    parser.add_argument("--env-id", default="MetaDrive-Tut-Easy-v0")
    parser.add_argument(
        "--log-dir",
        default="data/",
        type=str,
        help="The path of directory that you want to store the data to. "
             "Default: ./data/"
    )
    parser.add_argument("--max-steps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--start-steps", default=1e4, type=int)  # Time steps initial random policy is used

    parser.add_argument("--seed", default=0)
    parser.add_argument("--save_freq", default=2e3, type=int)  # How often (time steps) we save model
    parser.add_argument("--log_freq", default=1e3, type=int)  # How often (time steps) we print stats of model
    parser.add_argument("--explore_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates

    parser.add_argument("--load_model", action="store_true")
    args = parser.parse_args()

    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    trainer_path = os.path.join(log_dir, "td3")
    progress_path = os.path.join(log_dir, "td3", "progress.csv")

    if not os.path.exists(trainer_path):
        os.makedirs(trainer_path)

    environments = make_envs(
        env_id=args.env_id,
        log_dir=log_dir,
        num_envs=1,
        asynchronous=False,
    )
    env = environments.envs[0]

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Target policy smoothing is scaled wrt the action scale
    kwargs["policy_noise"] = args.policy_noise * max_action
    kwargs["noise_clip"] = args.noise_clip * max_action
    kwargs["policy_freq"] = args.policy_freq
    kwargs["lr"] = args.lr
    policy = TD3Trainer(**kwargs)

    discrete = False
    max_size = 1e-6
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    state, _ = env.reset()
    done = False

    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    # Setup some stats helpers
    log_count = 0
    reward_recorder = deque(maxlen=100)
    success_recorder = deque(maxlen=100)
    sample_timer = Timer()
    process_timer = Timer()
    update_timer = Timer()
    total_timer = Timer()
    progress = []
    loss_stats = {"target_q": np.nan, "actor_loss": np.nan, "critic_loss": np.nan}

    for t in tqdm.trange(args.max_steps, desc="Training Step"):
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_steps:
            action = env.action_space.sample()
        else:
            # TODO: Uncomment these lines and learn how TD3 generates exploratory actions.
            # action = (
            #         policy.select_action(np.array(state))
            #         + np.random.normal(0, max_action * args.explore_noise, size=action_dim)
            # ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        done_bool = float(done)  # if episode_timesteps < env._max_episode_steps else 0

        # if args.load_model:
        #     # Modify this to load proper models!
        #     policy.load(f"{log_dir}/models/default")

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_steps:
            loss_stats = policy.train(replay_buffer, args.batch_size)

        if done:
            reward_recorder.append(episode_reward)
            if "arrive_dest" in info:
                success_recorder.append(info.get("arrive_dest", 0))

            # Reset environment
            state, _ = env.reset()
            done = False

            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

            # ===== Log information =====
            if t - log_count * args.log_freq > args.log_freq:
                log_count = int(t // args.log_freq)
                stats = dict(
                    log_dir=log_dir,
                    frame_per_second=int(t / total_timer.now),
                    episode_reward=np.mean(reward_recorder),
                    total_steps=t,
                    total_episodes=episode_num,
                    total_time=total_timer.now,
                    **loss_stats
                )

                if success_recorder:
                    stats["success_rate"] = np.mean(success_recorder)

                progress.append(stats)
                pretty_print({
                    "===== TD3 Training Step {} =====".format(t): stats
                })

        if (t + 1) % args.save_freq == 0:
            policy.save(trainer_path)
            pd.DataFrame(progress).to_csv(progress_path)
            print("Trainer is saved at <{}>. Progress is saved at <{}>.".format(
                trainer_path, progress_path
            ))
