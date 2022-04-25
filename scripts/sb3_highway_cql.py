from email.policy import default
import gym
from custom_algorithm.cql import CQL
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, BaseCallback
import torch
from torch.nn import functional as F

import highway_env
import argparse
from functools import partial
import os

import wandb
from collections import defaultdict
import numpy as np
from tqdm import tqdm


def get_dataset(reward_scale=1.0, reward_shift=0.0, start_idx=0, end_idx=None, idx_diff=1, dataset_path=None, state_only=True, keys_to_remove=('states', 'info')):
    data_dict = defaultdict(list)
    if dataset_path is not None:
        for i in tqdm(range(start_idx, end_idx+1, idx_diff)):
            fname = os.path.join(dataset_path, 'rollouts_{}.npy'.format(i))
            if os.path.exists(fname):
                trajs = np.load(fname, allow_pickle=True)
                for dct in trajs:
                    for k, v in dct.items():
                        if k in keys_to_remove:
                            continue
                        if k in ['observations', 'next_observations']:
                            if state_only:
                                data_dict[k].extend([minidict['state'] for minidict in v])
                            else:
                                data_dict[k].extend([minidict['image'] for minidict in v])
                        else:
                            data_dict[k].extend(v)
            else:
                assert False, '{} does not exist'.format(fname)
        data_dict['terminals'] = data_dict['done'] # same as done
    else:
        assert False, 'dataset_path is None'
    
    for k in data_dict.keys():
        data_dict[k] = np.array(data_dict[k])
        if len(data_dict[k].shape) == 1:
            data_dict[k] = data_dict[k].reshape(-1, 1)
        print('{}: {}'.format(k, data_dict[k].shape))

    print ('Actions Max/Min/Mean: ', data_dict['actions'].max(), data_dict['actions'].min(), data_dict['actions'].mean())
    lim = 1 - 1e-5
    data_dict['actions'] = np.clip(data_dict['actions'], -lim, lim)
    data_dict['dones'] = data_dict['terminals']

    data_dict['rewards'] = (data_dict['rewards'] + reward_shift) * reward_scale

    print (data_dict.keys())
    return data_dict

TRAIN = True
if __name__ == '__main__':
    # Create the environment
    argparser = argparse.ArgumentParser(description='Train or test CQL model for highway-fast')
    argparser.add_argument('--env', type=str, help='environment ID', default='highway-v0')
    argparser.add_argument('--log_path', type=str, help='Logging Location', default='data/')
    argparser.add_argument('--no_wandb', action='store_true', help='Do not use wandb')
    argparser.add_argument('--batch_size', type=int, help='Batch size for training', default=256)
    argparser.add_argument('--discount', type=float, help='Learning rate', default=0.975)
    argparser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the network')
    argparser.add_argument('--layer_size', type=int, default=256, help='Number of layers in the network')
    argparser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')

    argparser.add_argument('--dataset_path', type=str, help='Dataset path', default='/home/asap7772/asap7772/highway_replay_buffers/highwayfast')
    # argparser.add_argument('--dataset_start_idx', type=int, help='Dataset start index', default=5000)
    argparser.add_argument('--dataset_end_idx', type=int, help='Dataset end index', default=100000)
    argparser.add_argument('--dataset_end_idx', type=int, help='Dataset end index', default=5000)
    argparser.add_argument('--dataset_idx_diff', type=int, help='Dataset index difference', default=5000)

    argparser.add_argument('--with_minq', action='store_false', help='With minq')
    argparser.add_argument('--min_q_weight', type=float, help='CQL Alpha', default=1.0)
    argparser.add_argument('--method', action='store_true', help='With method')
    argparser.add_argument('--method_temp', type=float, help='Temp', default=50.0)
    args = argparser.parse_args()


    data_to_load = get_dataset(reward_scale=1, reward_shift=0, start_idx=args.dataset_start_idx, end_idx=args.dataset_end_idx, idx_diff=args.dataset_idx_diff, dataset_path=args.dataset_path, state_only=True)
    args_dict = vars(args)
    
    str_args = ["_" + str(k) + '=' + str(v) for k, v in args_dict.items() if k not in ['log_path', 'no_wandb', 'dataset_path', 'dataset_start_idx', 'dataset_end_idx', 'dataset_idx_diff']]
    hyperparam_str = "".join(str_args)
    print(hyperparam_str)

    log_path = os.path.join(args.log_path, 'exp' + hyperparam_str +'/')
    
    env = gym.make(args.env)
    
    obs = env.reset()

    if not args.no_wandb:
        wandb.init(project="highway-env-rerun_v2", reinit=True, settings=wandb.Settings(start_method="fork"), config=args)
        wandb.run.name = log_path.split("/")[-2]
        wandb.run.save()
        
    eval_callback = EvalCallback(env, best_model_save_path=os.path.join(log_path, 'best_model'),
                             log_path=os.path.join(log_path, 'results'), eval_freq=500,
                             deterministic=True, render=False)
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_path)
    callback = CallbackList([checkpoint_callback, eval_callback])

    # Create the model
    net_arch = [args.layer_size] * args.num_layers
    print('net_arch', net_arch)
    
    model = CQL('MlpPolicy', env,
                policy_kwargs=dict(net_arch=net_arch),
                learning_rate=args.learning_rate,
                buffer_size=15000,
                batch_size=args.batch_size,
                gamma=args.discount,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                verbose=1,
                tensorboard_log=log_path,
                with_minq=args.with_minq,
                min_q_weight=args.min_q_weight,
                wandb_log=not args.no_wandb,
                method=args.method,
                method_temp=args.method_temp,
    )

    # Setup Replay Buffer
    model.replay_buffer.handle_timeout_termination = False
    for i in tqdm(range(data_to_load['observations'].shape[0])):
        model.replay_buffer.add(
            data_to_load['observations'][i],
            data_to_load['next_observations'][i], 
            data_to_load['actions'][i], 
            data_to_load['rewards'][i], 
            data_to_load['done'][i],
            {},
        )
    model.replay_buffer.handle_timeout_termination = True

    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(1e6), callback=callback)
        model.save(os.path.join(log_path, "model"))
        del model

    # Run the trained model and record video
    model = CQL.load(os.path.join(log_path, "model"), env=env)
    env = RecordVideo(env, video_folder="racetrack_ppo/videos", episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)
    env.configure({"simulation_frequency": 15})  # Higher FPS for rendering

    for videos in range(10):
        done = False
        obs = env.reset()
        while not done:
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, info = env.step(action)
            # Render
            env.render()
    env.close()
