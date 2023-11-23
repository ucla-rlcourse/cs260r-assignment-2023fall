
# If I want to train an agent from scratch:

python train_ppo_in_multiagent_env.py \
--log-dir train_ppo_in_multiagent_env \
--num-processes 10 \
--num-steps 4_000 \
--max-steps 10_000_000


# If I want to use a pretrained model:

#python train_ppo_in_multiagent_env.py \
#--log-dir train_ppo_in_multiagent_env_nocrashreward \
#--num-processes 10 \
#--num-steps 4_000 \
#--pretrained-model-suffix iter275 \
#--pretrained-model-log-dir agents/example_agent \
#--max-steps 10_000_000
