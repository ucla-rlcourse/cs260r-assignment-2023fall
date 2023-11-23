
# Assignment 3: Reinforcement Learning Pipeline in Practice


*CS260R 2023Fall: Reinforcement Learning. Department of Computer Science at University of California, Los Angeles.
Course Instructor: Professor Bolei ZHOU. Assignment author: Zhenghao (Mark) PENG, Yuxin Liu.*

---


> NOTE: **We only grade your assignment based on the performance of your agent and the report file.**


## Overview

The ultimate goal of this assignment is to use all knowledge you learned to train a competitive RL agent that can win the tournament.

We will focus on the MetaDrive Racing environment which has a long track and many turnouts. The environment supports 12+ agents running concurrently, but we will hold the competition with two agents.


Your job is to prepare a subfolder inside `agents/` folder that hosts your agent. Each student will submit one RL agent to participate the tournament.

You are free to use arbitrary algorithm and codebase to implement your agent. You don't even need to run the code we provide here in `train_ppo_in_multiagent_env.py` and `train_ppo_in_singleagent_env.py`. The only thing you need to ensure is that your agent can be properly packed into the `Policy` class and can be evaluated and ran by us without bug.


Optionally, you can also submit a PDF file that describes your name & UID as well as the effort and innovation you have for training the agent. Please put the PDF file into your submission as well as submit it to gradescope. But you can also choose not to prepare the PDF file. It's totally up to you.



## Environment overview

The MetaDrive racing environment contains a long two-lane track with no intersection, no traffic vehicles and with a high wall that blocks vehicles driving out of the road.

The environment is natively a multi-agent RL environment. The input and output data is all in the form of Python dicts, whose keys are the agents' name such as `agent0`, `agent1`, ..., and values are the corresponding information.

In `train_ppo_in_multiagent_env.py` and `eval_single_agent_env.py`, we create a wrapper of the environment that sets `config["num_agents"] = 1` and wraps and unwraps the data passing in
and out from the environment. Therefore, the environment behaves like a single-agent RL environment and we can reuse the single-agent RL algorithm such as PPO to train agent in this environment.

In `train_ppo_in_multiagent_env.py`, we create a wrapper that still makes the environment behaves like a single-agent env, but this time we can load a trained agent serving as the `agent1` and let the learning agent to control `agent0`.


Remind that we only require you to submit one agent and there is no requirement on which environment and which algorithm you need to use. You can choose not to use any training script we provide and implement your own agent even in other algorithms.



## Evaluation protocol & Grading


### Qualification Stage

In qualification stage, we will evaluate each submission and get a ranking based on the single-agent performance. The top 64 agents will participate the tournament.


We will iterate over all submissions and run the single agent evaluation similar to the one in `eval_single_agent_env.py`. This evaluation is conducted in the single-agent environment.


We will use the longitudinal movement of your agent travels in one episode as the metric. Running your agent in the environment, we know that there is a start point and an end point in your agent's trajectory. Consider there is an underlying central line in the track, we can project these two points into that central line. The longitudinal movement is defined as the distance between two projected points in that central line. We modify the reward function in the eval environment so that `episode_reward` reflects the longitudinal movement. We will use `episode_reward` reported by `eval_single_agent_env.py` as the metric.


We will run each agent for 100 episodes and use result averaged over these 100 episodes as the metric of your agent.

It is possible that an agent can achieve 100% success rate. If multiple submissions achieve 100% success rate, to avoid the wrong ranking caused by the numerical instability in the `episode_reward`, we will consider the one with smaller average episode length to be better.
(But I think the probability is quite low that more than 64 agents achieve 100% success rate :) )


**The top 64 submissions will participate the tournament.**


### Tournament


In the tournament, we will hold a series of head-to-head matches between agents. All survived agents will randomly form a set of two-agent matches. We will run 11 episodes for each match. The agent wins in more episodes will enter the next round.


We will iterate over all submissions and run the multi-agent evaluation similar to the one in `eval_single_agent_env.py`. The winner of an episode is defined as:

* If one agent reaches the destination and one not, the one reaches the destination is the winner in this episode. (**This is called a clear win.**)
* If both reach the destination, the one that is faster is the winner.
* If both fail to reach the destination, the one that has larger longitudinal movement wins.

We will start from forming 32 matches between these 64 agents. The 32 winners will enter next round. We repeat this process until there is only one agent.

**We will invite the creator of the top 1 agent to share how they/he/she creates the agent.**



### Party Time

The MetaDrive Racing environment can support up to 12+ agents running concurrently. We will host a match for the top-8 agents and see what happens. This is not graded and is just for fun.

### Grading scheme

+90 score will be given if your agent is successfully evaluated by TA without bug in the qualification stage. 

For the agents that enter the tournament, each time it wins a match (that is it has more wins in 11 episodes of evaluation), the creator will have +5 score. 

We also define the concept of **clear win**: if your agent reaches the destination while your component does not, this is called a clear win and for this particular episode you will win +0.5 score. This rule is designed to encourage more competitive agent.

We will also give up to +50 bonus for any innovation and interesting efforts you've done to create the agent. This is not related to the performance of your agent, everybody can win it. We welcome any kind of innovation, experiments (even they might fail) and exploration! **You can optionally include a PDF file in any format, style and template to describe your efforts. If you decide to write this file, include it in the submission folder as well as submit it to gradescope. But you don't need to submit it if you decide not to write it.**

So theoretically the top-1 winner can earn as many score as:

```
+ 90 (basic score) 
+ 5 * 6 (win 6 matches)
+ 0.5 * 11 * 6 (clear wins in each episode)
+ 50 (innovation bonus)
= 153 + 50
= 203
```


## Submission


### The policy interface

You can check out the folder `agents/example_agent` to see how we create a self-contained agent. Specifically, you have to implement `agents/youragentname/agent.py` file with a class `Policy` in it. The `Policy` class need to implement the `__call__` function. You can use relative import to import files in your folder but please do not import file outside your submission folder because I will not able to import that file. If you want to use external package, please try to minimize the dependency to it and tell me how can I install the package in my python environment.

Here are three fields you need to fill. First, the subfolder name, e.g. `example_agent`, is your **agent name**. In the `Policy` class in `agent.py`, you need to fill the class variables `CREATOR_NAME` and `CREATOR_UID`. You can choose arbitrary agent name. Make sure the agent name (the subfolder name) is not started with number.

We will put all submissions (the subfolders) into my `agents/` folder. A script will automatically register the `Policy` in `agents.py` in each subfolder.


### Deliverable

Please compress your subfolder into a zip file and submit it to bruinlean. Please make sure that when uncompress your ZIP file, we will have these file structure:

```
- agent_name # the subfolder name
  - agent.py
  - checkpoint...pkl  # Some sort of a checkpoint file. You don't have to use the provided code.
  - ...
```

The point is you need to make sure the subfolder is there inside your ZIP (otherwise I don't know your agent name). For example, this submission is invalid:

```
- agent.py
- checkpoint...pkl  # Some sort of a checkpoint file. You don't have to use the provided code.
- ...
```

You can optionally add a PDF file inside your subfolder to describe your innovation, efforts and insights when developing the agent. The PDF file can be in any form, contain anything. Please also submit the PDF file to gradescope. You don't have to write it if you don't want. Submitting a PDF file does not guarentee bonus.




### Local test before submission


You can evaluate your agent with this command:

```bash

python eval_single_agent_env.py --agent-name YOUR-AGENT-NAME

# To test the code before training, you can try to run TA's agent:
# NOTE THAT I deleted my checkpoint and comment out the line that loads weights
# in the demo file. You need to modify the code to load your agent properly.
python eval_single_agent_env.py --agent-name example_agent
```

If your agent is evaluated successfully and the reported performance looks good to you, then your submission is prepared well.


This script will report the performance of your agent in the single agent racing environment. The same script will be used to determine whether your agent is qualified to participate the tournament.


## Provided code

Again, you can use arbitrary code, algorithm, and model. 

We provide the training script to train a PPO agent in single-agent environment. Please check out `train_ppo_singleagent_env.py` and `train_ppo_singleagent_env.sh`. We also provide a script to train a RL agent in an environment with multiple agents. That is, you can train an RL agent in a competitive environment where there exists another agent that can interact with yours. Please check out `train_ppo_in_multiagent_env.py` and `train_ppo_in_multiagent_env.sh`.

We provide an example agent in `agents/`. Unfortunately, I can not provide my trained checkpoint there and I commented out the line to restore checkpoint.

We also provide two evaluation scripts. You can check out `eval_multi_agent_performance.py` and `eval_single_agent_performance.sh`.

Notes:

* You can set `--num-processes=1` to avoid using multiprocessing. This helps debugging.
* You can load pretrained model in training via flag `--pretrained-model-log-dir` and `--pretrained-model-suffix`.
* You can set the opponent agent when training in multi-agent env via `--opponent-agent-name`.