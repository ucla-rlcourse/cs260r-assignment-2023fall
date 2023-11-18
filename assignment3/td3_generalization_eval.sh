# Example script to run a batch of TD3 generalization experiments
for num in 1 5 10 20 50 100; do
  python eval_td3.py \
  --log-dir MetaDrive-Tut-${num}Env-v0/td3 \
  --num-envs 10 \
  --num-episodes 100
done
