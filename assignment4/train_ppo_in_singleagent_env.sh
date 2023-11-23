# If I want to train an agent from scratch:

python train_ppo_in_singleagent_env.py \
--log-dir train_ppo_in_single_agent_env \
--num-processes 10 \
--num-steps 4_000 \
--max-steps 10_000_000
