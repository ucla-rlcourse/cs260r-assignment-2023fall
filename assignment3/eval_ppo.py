import argparse

import pandas as pd

from core.envs import make_envs
from vis import evaluate_in_batch, PPOTrainer, PPOConfig

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

    config = PPOConfig()

    trainer = PPOTrainer(envs, config)

    trainer.load_w(log_dir, suffix="final")


    def _policy(obs):
        values, actions, log_probs = trainer.compute_action(obs)
        return actions.cpu().numpy()


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
        "PPO", num_episodes, env_id, eval_reward, path
    ))
