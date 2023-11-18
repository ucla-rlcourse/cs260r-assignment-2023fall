# Example script to run a batch of PPO generalization experiments
for num in 1 5 10 20 50 100; do
  python eval_ppo.py \
  --log-dir MetaDrive-Tut-${num}Env-v0/ppo \
  --num-envs 10 \
  --num-episodes 100
done
