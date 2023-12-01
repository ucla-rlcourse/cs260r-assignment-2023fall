"""
This script evaluates the performance of multiple agents running concurrently in the same environment. Similar to the
single-agent evaluation, we set the reward function to report the longitudinal movement. We define a Policy Map in
the beginning of the file to determine what agents to use.
"""
import argparse
from collections import defaultdict

import numpy as np
import torch
import tqdm
from metadrive.envs.marl_envs.marl_racing_env import MultiAgentRacingEnv

from agents import load_policies
from core.ppo_trainer import PPOConfig
from core.utils import pretty_print

POLICY_MAP = {
    "example_agent": "agent0",
    "example_agent_2": "agent1",
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num-processes",
    default=10,
    type=int,
    help="The number of parallel RL environments. Default: 10"
)
parser.add_argument(
    "--num-episodes",
    default=11,
    type=int,
    help="The number of episode to evaluate. Default: 11"
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
parser.add_argument(
    "--generate-video",
    action="store_true",
    help="Whether to generate videos when rendering."
)
args = parser.parse_args()


class MultiAgentRacingEnvWithSimplifiedReward(MultiAgentRacingEnv):
    """We do not wrap the environment as a single-agent env here."""

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

    generate_video = args.generate_video

    config = PPOConfig()

    config.num_processes = args.num_processes

    # Seed the environments and setup torch
    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_num_threads(1)

    render = args.render
    video_bev = []
    video_interface = []

    # Create environment. Here we don't use the vectorized environment wrapper.
    env = MultiAgentRacingEnvWithSimplifiedReward(dict(

        # We only evaluate 2 agents now.
        num_agents=2,

        # If you want to extend the number of agents to 12, you need to fill the policies for agent2, ..., agent11 and
        # use these configs:
        # num_agents=12,
        # map_config=dict(lane_num=2, exit_length=60),  # Increasing the space in spawning the agents.

        # If you are using Mac or headless machine and want to render the top-down view only, comment out next line.
        use_render=render or generate_video,

        # This config sets the color of agent0 to green.
        target_vehicle_configs=dict(
            agent0=dict(use_special_color=True)
        )

    ))

    total_episodes_to_eval = args.num_episodes

    # Instantiate all policies here.
    all_policies = load_policies()
    policy_map = {}
    for policy_name, Policy in all_policies.items():
        if policy_name in POLICY_MAP:
            control_agent_name = POLICY_MAP[policy_name]
            policy = Policy()
            if hasattr(policy, "reset"):
                policy.reset()
            policy_map[control_agent_name] = Policy()

    print("==================================================")
    print(f"EVALUATING AGENTS {policy_map.keys()} IN MULTI-AGENT ENVIRONMENT.")
    print("==================================================")

    # Setup some stats helpers
    total_episodes = total_steps = iteration = 0
    last_total_steps = 0
    count = 0

    result_recorder = defaultdict(lambda: defaultdict(list))  # A nested dict

    print("Start evaluation!")
    try:
        obs_dict, _ = env.reset()
        terminated_dict = {}
        with tqdm.tqdm(total=int(total_episodes_to_eval)) as pbar:
            while True:

                # Form the action dict
                action_dict = {}
                for agent_name, agent_obs in obs_dict.items():
                    if agent_name in terminated_dict and terminated_dict[agent_name]:
                        continue
                    act = policy_map[agent_name](agent_obs)
                    if act.ndim == 2:
                        act = np.squeeze(act, axis=0)
                    action_dict[agent_name] = act

                # Step the environment
                obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = env.step(action_dict)

                count += 1

                if render and not generate_video:
                    env.render(mode="topdown")

                if generate_video:
                    import pygame

                    img_interface = env.render(mode="rgb_array")
                    img_bev = env.render(
                        mode="topdown",
                        # show_agent_name=True,
                        # target_vehicle_heading_up=True,
                        draw_target_vehicle_trajectory=False,
                        film_size=(5000, 5000),
                        screen_size=(1000, 1000),
                        crash_vehicle_done=False,
                    )
                    img_bev = pygame.surfarray.array3d(img_bev)
                    img_bev = img_bev.swapaxes(0, 1)
                    # img_bev = img_bev[::-1]
                    video_bev.append(img_bev)
                    video_interface.append(img_interface)

                for agent_name, agent_done in terminated_dict.items():
                    if agent_name == "__all__":
                        continue
                    agent_info = info_dict[agent_name]
                    if "crash_vehicle" in agent_info:
                        result_recorder[agent_name]["crash_vehicle_rate"].append(agent_info["crash_vehicle"])
                    if "crash_sidewalk" in agent_info:
                        result_recorder[agent_name]["crash_sidewalk_rate"].append(agent_info["crash_sidewalk"])
                    if "idle" in agent_info:
                        result_recorder[agent_name]["idle_rate"].append(agent_info["idle"])
                    if "speed_km_h" in agent_info:
                        result_recorder[agent_name]["speed_km_h"].append(agent_info["speed_km_h"])
                    if agent_done:  # the episode is done
                        # Record the reward of the terminated episode to
                        result_recorder[agent_name]["episode_reward"].append(agent_info["episode_reward"])
                        if "arrive_dest" in agent_info:
                            result_recorder[agent_name]["success_rate"].append(agent_info["arrive_dest"])
                        if "max_step" in agent_info:
                            result_recorder[agent_name]["max_step_rate"].append(agent_info["max_step"])
                        if "episode_length" in agent_info:
                            result_recorder[agent_name]["episode_length"].append(agent_info["episode_length"])

                if terminated_dict["__all__"]:
                    total_episodes += 1
                    obs_dict, _ = env.reset()
                    terminated_dict = {}

                    for policy in policy_map.values():
                        if hasattr(policy, "reset"):
                            policy.reset()

                pbar.update(total_episodes - pbar.n)
                if total_episodes >= total_episodes_to_eval:
                    break
    finally:
        env.close()


    print("==================================================")
    print("AVERAGE PERFORMANCE")
    stat = {}
    for policy_name, control_agent_name in POLICY_MAP.items():
        agent_result_recorder = result_recorder[control_agent_name]
        agent_name = f"{control_agent_name} ({policy_name})"
        stat[agent_name] = {k: np.mean(v) for k, v in agent_result_recorder.items()}
    pretty_print(stat)
    print("==================================================")

    print("==================================================")
    print("RESULT:")
    win_stat = {agent_name: 0 for agent_name in result_recorder.keys()}
    clear_win_stat = {agent_name: 0 for agent_name in result_recorder.keys()}
    score_stat = {agent_name: 0 for agent_name in result_recorder.keys()}
    assert len(win_stat) == 2, "Only support 2 agents for now."
    for episode_count in range(total_episodes_to_eval):
        a0_succ = result_recorder["agent0"]["success_rate"][episode_count]
        a1_succ = result_recorder["agent1"]["success_rate"][episode_count]

        # Only one agent arrives destination:
        if np.logical_xor(a0_succ, a1_succ).item():
            if a0_succ:
                win_stat["agent0"] += 1
                clear_win_stat["agent0"] += 1
                score_stat["agent0"] += 2
            else:
                win_stat["agent1"] += 1
                clear_win_stat["agent1"] += 1
                score_stat["agent1"] += 2

        # If both arrive destination: faster agent winds.
        if a0_succ and a1_succ:
            a0_len = result_recorder["agent0"]["episode_length"][episode_count]
            a1_len = result_recorder["agent1"]["episode_length"][episode_count]
            if a0_len < a1_len:
                win_stat["agent0"] += 1
                score_stat["agent0"] += 1
            elif a0_len > a1_len:
                win_stat["agent1"] += 1
                score_stat["agent1"] += 1
            else:  # If both arrive destination at the same time: higher speed agent wins.
                # I don't believe this case will happen, but just write the code to cover this.
                if result_recorder["agent0"]["speed_km_h"][episode_count] > \
                        result_recorder["agent1"]["speed_km_h"][episode_count]:
                    win_stat["agent0"] += 1
                    score_stat["agent0"] += 1
                else:
                    win_stat["agent1"] += 1
                    score_stat["agent1"] += 1

        # If both fails to arrive destination: higher longitude movement agent wins.
        if not (a0_succ or a1_succ):
            a0_rew = result_recorder["agent0"]["episode_reward"][episode_count]
            a1_rew = result_recorder["agent1"]["episode_reward"][episode_count]
            if a0_rew > a1_rew:
                win_stat["agent0"] += 1
                score_stat["agent0"] += 1
            else:
                win_stat["agent1"] += 1
                score_stat["agent1"] += 1

    win_stat_print = {}
    for policy_name, control_agent_name in POLICY_MAP.items():
        wins = win_stat[control_agent_name]
        agent_name = f"{control_agent_name} ({policy_name})"
        win_stat_print[agent_name] = {
            "wins": wins,
            "win_rate": wins / total_episodes_to_eval,
            "clear_wins": clear_win_stat[control_agent_name],
            "score": score_stat[control_agent_name]
        }
    pretty_print(win_stat_print)
    inverse_policy_map = {v: k for k, v in POLICY_MAP.items()}
    if win_stat["agent0"] > win_stat["agent1"]:
        print(
            f"Agent0 {inverse_policy_map['agent0']} (CREATOR: {policy_map['agent0'].CREATOR_NAME}, UID: {policy_map['agent0'].CREATOR_UID}) Wins!")
    else:
        print(
            f"Agent1 {inverse_policy_map['agent1']} (CREATOR: {policy_map['agent1'].CREATOR_NAME}, UID: {policy_map['agent1'].CREATOR_UID}) Wins!")
    print("==================================================")


    if generate_video:
        import datetime
        import os
        import mediapy

        FPS = 60

        folder_name = "marl_racing_video_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(folder_name, exist_ok=True)

        video_base_name = f"{folder_name}/video"
        video_name_bev = video_base_name + "_bev.mp4"
        print("BEV video should be saved at: ", video_name_bev)
        mediapy.write_video(video_name_bev, video_bev, fps=FPS)

        video_name_interface = video_base_name + "_interface.mp4"
        print("Interface video should be saved at: ", video_name_interface)
        mediapy.write_video(video_name_interface, video_interface, fps=FPS)
