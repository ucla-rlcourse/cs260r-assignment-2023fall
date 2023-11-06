import argparse
import os

import pandas as pd

from core.envs import make_envs
from core.td3_trainer import TD3Trainer
from vis import evaluate_in_batch

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
    help="The number of parallel environments for evaluation. Default: 10"
)
parser.add_argument(
    "--seed",
    default=0,
    type=int,
    help="The random seed. Default: 0"
)
parser.add_argument(
    "--num-episodes",
    default=50,
    type=int,
)
parser.add_argument(
    "--env-id",

    # === See here! We will use test environment for eval! ===
    default="MetaDrive-Tut-Test-v0",

    type=str,
)

if __name__ == '__main__':
    args = parser.parse_args()

    log_dir = args.log_dir
    num_episodes = args.num_episodes
    num_envs = args.num_envs
    env_id = args.env_id

    if "MetaDrive" in env_id:
        from core.utils import register_metadrive

        register_metadrive()

    envs = make_envs(
        env_id=env_id,
        log_dir=log_dir,
        num_envs=num_envs,
        asynchronous=True,
    )

    # env = envs.envs[0]

    state_dim = envs.observation_space.shape[0]
    action_dim = envs.action_space.shape[0]
    max_action = float(envs.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
    }
    trainer = TD3Trainer(**kwargs)

    trainer.load(os.path.join(log_dir))


    def _policy(obs):
        return trainer.select_action_in_batch(obs)


    eval_reward, eval_info = evaluate_in_batch(
        policy=_policy,
        envs=envs,
        num_episodes=num_episodes,
    )

    df = pd.DataFrame({"rewards": eval_info["rewards"], "successes": eval_info["successes"]})
    path = "{}/eval_results.csv".format(log_dir)
    df.to_csv(path)

    print("The average return after running {} agent for {} episodes in {} environment: {}.\n" \
          "Result is saved at: {}".format(
        "TD3", num_episodes, env_id, eval_reward, path
    ))
