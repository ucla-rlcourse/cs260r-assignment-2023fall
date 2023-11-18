# Example script to run a batch of PPO generalization experiments

# 1M steps is enough.

for num in 1 5 10 20 50 100; do
  python train_ppo.py \
  --env-id MetaDrive-Tut-${num}Env-v0 \
  --log-dir MetaDrive-Tut-${num}Env-v0 \
  --num-envs 10 \
  --max-steps 1_000_000
done


# Run this command to overwatch the training progress:
# watch tail -n 5 ppo*train.log
