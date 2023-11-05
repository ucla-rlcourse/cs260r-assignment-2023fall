# Example script to run a batch of PPO generalization experiments
for num in 1 5 10 20 50 100; do
  python train_td3.py \
  --env-id MetaDrive-Tut-${num}Env-v0 \
  --log-dir MetaDrive-Tut-${num}Env-v0 \
  --max-steps 500000 > td3_metadrive_${num}env.log 2>&1 &
done


# Run this command to overwatch the training progress:
# watch tail -n 5 *.log
