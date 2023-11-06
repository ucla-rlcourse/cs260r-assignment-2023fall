# Example script to run a batch of PPO generalization experiments

for num in 1 5 10 20 50 100; do
  python train_gail.py \
  --env-id MetaDrive-Tut-${num}Env-v0 \
  --log-dir MetaDrive-Tut-${num}Env-v0 \
  --num-envs 10 \
  --max-steps 1000000 \
  --expert-dataset-size 30000 \
  > gail_metadrive_${num}env_train.log 2>&1 &
done


# Run this command to overwatch the training progress:
# watch tail -n 5 gail*train.log
