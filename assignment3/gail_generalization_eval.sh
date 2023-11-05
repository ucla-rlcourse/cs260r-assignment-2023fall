# Example script to run a batch of GAIL generalization experiments
for num in 1 5 10 20 50 100; do
  python eval_gail.py \
  --log-dir MetaDrive-Tut-${num}Env-v0/gail \
  --num-envs 10 \
  --num-episodes 100 > gail_metadrive_${num}env_eval.log 2>&1 &
done
