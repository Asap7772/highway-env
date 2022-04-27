export CUDA_VISIBLE_DEVICES=0
python /home/asap7772/asap7772/highway-env/scripts/sb3_highway_cql.py --env highway-fast-v0 --min_q_weight 0.01 &

export CUDA_VISIBLE_DEVICES=1
python /home/asap7772/asap7772/highway-env/scripts/sb3_highway_cql.py --env highway-fast-v0 --min_q_weight 0.1 &

export CUDA_VISIBLE_DEVICES=2
python /home/asap7772/asap7772/highway-env/scripts/sb3_highway_cql.py --env highway-fast-v0 --min_q_weight 0.2 &

export CUDA_VISIBLE_DEVICES=3
python /home/asap7772/asap7772/highway-env/scripts/sb3_highway_cql.py --env highway-fast-v0 --min_q_weight 0.5 &

export CUDA_VISIBLE_DEVICES=4
python /home/asap7772/asap7772/highway-env/scripts/sb3_highway_cql.py --env highway-fast-v0 --min_q_weight 1 &

export CUDA_VISIBLE_DEVICES=5
python /home/asap7772/asap7772/highway-env/scripts/sb3_highway_cql.py --env highway-fast-v0 --min_q_weight 2 &

export CUDA_VISIBLE_DEVICES=6
python /home/asap7772/asap7772/highway-env/scripts/sb3_highway_cql.py --env highway-fast-v0 --min_q_weight 5 &

export CUDA_VISIBLE_DEVICES=7
python /home/asap7772/asap7772/highway-env/scripts/sb3_highway_cql.py --env highway-fast-v0 --min_q_weight 10 &