"""
Env Configuration
"""

import os
from packaging import version

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb

from email.policy import default
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, BaseCallback
import torch
from torch.nn import functional as F
from tensorflow.python.summary.summary_iterator import summary_iterator
from collections import defaultdict
import numpy as np
import pprint
from tqdm import tqdm

import highway_env

major_ver, minor_ver, _ = version.parse(tb.__version__).release
assert major_ver >= 2 and minor_ver >= 3, \
    "This notebook requires TensorBoard 2.3 or later."
print("TensorBoard version: ", tb.__version__)


"""
Env Configuration # need to check if different from image env
"""
os.environ["OFFSCREEN_RENDERING"] = "1"
env_name='highway-fast-v0'
env = gym.make(env_name)
eval_env = gym.make(env_name)
pprint.pprint(env.config)

"""
Load Tensorboard data
"""
path = '/home/asap7772/highway-env/data'
des_path_snippet = 'exp_env=highway-fast-v0_batch_size=256_discount=0.9_exploration_fraction=0.95_num_layers=4_layer_size=256_learning_rate=0.001_state=False'
output_path = '/home/asap7772/highway-env/replay_buffers/'

lst = []
for root, dirs, files in os.walk(path, topdown=False):
    for name in files:
      x = os.path.join(root, name)
      if 'tfevents' in x:
        lst.append(x)
        
found = None
for x in lst:
  if des_path_snippet in x:
      found = x
      break

def get_data(path, prefix=''):
    data = defaultdict(list)
    for e in summary_iterator(path):
        for v in e.summary.value:
            data[v.tag].append(v.simple_value)
    return data

dct = get_data(found)
return_vals = np.array(dct['rollout/ep_rew_mean'])

print("Successfully loaded data from {}".format(found))
print("Return Stats:")
print("Mean", return_vals.mean())
print("Max", return_vals.max())
print("Min", return_vals.min())
print("Std", return_vals.std())
    
"""
Get Model
"""

path_split = found.split('/')
dir = '/'.join(path_split[:-2])

model_paths = []
how_often = 5000
for x in sorted(os.listdir(dir)):
  if 'rl_model_' in x and '_steps' in x:
    which_model = int(x.split('_')[-2])
    if which_model % how_often == 0:
      model_paths.append((which_model, os.path.join(dir, x)))

model_paths = sorted(model_paths, key=lambda x: x[0])


def rollouts(model, env_name='highway-v0', n_rollouts=200, max_steps=1000):
  trajs = []
  returns = []
  i = 0
  for i in tqdm(range(n_rollouts)):
    env = gym.make(env_name)
    obs = env.reset()
    done = False
    traj = defaultdict(list)
    while not done:
      action, _states = model.predict(obs)
      next_obs, rew, done, _info = env.step(action.item())
      image = env.render(mode="rgb_array")
      
      obs_full = {'image': image, 'state': obs}
      
      traj['observations'].append(obs_full)
      traj['next_observations'].append(next_obs)
      traj['actions'].append(action)
      traj['rewards'].append(rew)
      traj['done'].append(done)
      traj['states'].append(_states)
      traj['info'].append(_info)
      
      obs = next_obs
      i += 1
      
      if i > max_steps:
        break

    returns.append(sum(traj['rewards']))
    trajs.append(traj)
  
  returns_np = np.array(returns)
  return_statistics = {
    'mean': returns_np.mean(),
    'std': returns_np.std(),
    'min': returns_np.min(),
    'max': returns_np.max(),
  }
  print("Return Stats:", return_statistics)
  return trajs

for which, model_path in model_paths:
  model = DQN.load(model_path, env=env)
  rs = rollouts(model, env_name=env_name, n_rollouts=200, max_steps=1000)
  output_dir = os.path.join(output_path, ''.join(env_name.split('-')[:-1]))
  os.makedirs(output_dir, exist_ok=True)
  full_path = os.path.join(output_dir, 'rollouts_{}'.format(which))
  np.save(full_path, rs, allow_pickle=True)
  print("Saved rollouts for epoch {} at {}".format(which, full_path))

  
  
  
