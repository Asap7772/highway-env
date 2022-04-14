export CUDA_VISIBLE_DEVICES=0
python /home/asap7772/highway-env/scripts/sb3_highway_dqn_cnn.py --env highway-v0

export CUDA_VISIBLE_DEVICES=1
python /home/asap7772/highway-env/scripts/sb3_highway_dqn_cnn.py --env merge-v0

export CUDA_VISIBLE_DEVICES=2
python /home/asap7772/highway-env/scripts/sb3_highway_dqn_cnn.py --env roundabout-v0

export CUDA_VISIBLE_DEVICES=3
python /home/asap7772/highway-env/scripts/sb3_highway_dqn_cnn.py --env intersection-v0

export CUDA_VISIBLE_DEVICES=0
python /home/asap7772/highway-env/scripts/sb3_highway_dqn.py --env highway-v0

export CUDA_VISIBLE_DEVICES=1
python /home/asap7772/highway-env/scripts/sb3_highway_dqn.py --env merge-v0

export CUDA_VISIBLE_DEVICES=2
python /home/asap7772/highway-env/scripts/sb3_highway_dqn.py --env roundabout-v0

export CUDA_VISIBLE_DEVICES=3
python /home/asap7772/highway-env/scripts/sb3_highway_dqn.py --env intersection-v0

export CUDA_VISIBLE_DEVICES=0
python /home/asap7772/highway-env/scripts/sb3_highway_dqn_cnn.py --env highway-fast-v0

export CUDA_VISIBLE_DEVICES=1
python /home/asap7772/highway-env/scripts/sb3_highway_dqn.py --env highway-fast-v0