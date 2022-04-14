from email.policy import default
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, BaseCallback
import torch
from torch.nn import functional as F

import highway_env
import argparse
from functools import partial
import os

import wandb


TRAIN = True
if __name__ == '__main__':
    # Create the environment
    
    argparser = argparse.ArgumentParser(description='Train or test DQN model for highway-fast')
    argparser.add_argument('--env', type=str, help='environment ID', default='highway-v0')
    argparser.add_argument('--log_path', type=str, help='Logging Location', default='data/')
    argparser.add_argument('--batch_size', type=int, help='Batch size for training', default=256)
    argparser.add_argument('--discount', type=float, help='Learning rate', default=0.975)
    argparser.add_argument('--exploration_fraction', type=float, help='Exploration fraction', default=0.7)
    argparser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the network')
    argparser.add_argument('--layer_size', type=int, default=256, help='Number of layers in the network')
    argparser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    argparser.add_argument('--state', action='store_true', help='Dummy for wandb')
    args = argparser.parse_args()

    args_dict = vars(args)
    
    str_args = ["_" + str(k) + '=' + str(v) for k, v in args_dict.items() if k not in ['log_path']]
    hyperparam_str = " ".join(str_args)
    print(hyperparam_str)

    log_path = os.path.join(args.log_path, 'exp' + hyperparam_str +'/')
    
    env = gym.make(args.env)
    
    obs = env.reset()
    wandb.init(project="highway-env-rerun", reinit=True, settings=wandb.Settings(start_method="fork"), config=args)
    wandb.run.name = log_path.split("/")[-2]
    wandb.run.save()

    class TensorboardCallback(BaseCallback):
        """
        Custom callback for plotting additional values in tensorboard.
        """

        def __init__(self, verbose=0):
            super(TensorboardCallback, self).__init__(verbose)

        def _on_step(self) -> bool:
            # Log scalar value (here a random variable)
            buffer = model.replay_buffer
            num = buffer.buffer_size if buffer.full else buffer.pos
            
            if num < args.batch_size:
                return True
            
            replay_data = buffer.sample(args.batch_size, env=model._vec_normalize_env) # env doesn't have normalize_obs?
            
            with torch.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = model.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * model.gamma * next_q_values
            
            current_q_values = model.q_net(replay_data.observations)
            current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            
            # Log scalar value
            self.logger.record('qf1 loss', loss.item())
            self.logger.record('qf1 mean', current_q_values.mean().item())
            self.logger.record('qf1 std', current_q_values.std().item())
            self.logger.record('qf1 max', current_q_values.max().item())
            self.logger.record('qf1 min', current_q_values.min().item())
            
            self.logger.record('target qf1 mean', target_q_values.mean().item())
            self.logger.record('target qf1 std', target_q_values.std().item())
            self.logger.record('target qf1 max', target_q_values.max().item())
            self.logger.record('target qf1 min', target_q_values.min().item())
            
            self.logger.record('rewards mean', replay_data.rewards.mean().item())
            self.logger.record('rewards std', replay_data.rewards.std().item())
            self.logger.record('rewards max', replay_data.rewards.max().item())
            self.logger.record('rewards min', replay_data.rewards.min().item())
            
            for x in self.logger.name_to_value:
                wandb.log({x: self.logger.name_to_value[x]})
            return True
        
    eval_callback = EvalCallback(env, best_model_save_path=os.path.join(log_path, 'best_model'),
                             log_path=os.path.join(log_path, 'results'), eval_freq=500,
                             deterministic=True, render=False)
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_path)
    callback = CallbackList([checkpoint_callback, eval_callback, TensorboardCallback()])

    # Create the model
    net_arch = [args.layer_size] * args.num_layers
    print('net_arch', net_arch)
    
    model = DQN('MlpPolicy', env,
                policy_kwargs=dict(net_arch=net_arch),
                learning_rate=args.learning_rate,
                buffer_size=15000,
                learning_starts=200,
                batch_size=args.batch_size,
                gamma=args.discount,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                exploration_fraction=args.exploration_fraction,
                verbose=1,
                tensorboard_log=log_path
    )

    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(1e5), callback=callback)
        model.save(os.path.join(log_path, "model"))
        del model

    # Run the trained model and record video
    model = DQN.load(os.path.join(log_path, "model"), env=env)
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
