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


def train_env(env_name) -> gym.Env:
    env = gym.make(env_name)
    env.configure({
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "scaling": 1.75,
        },
    })
    env.reset()
    return env


def test_env(env_name) -> gym.Env:
    env = train_env(env_name)
    env.configure({"policy_frequency": 15, "duration": 20 * 15})
    env.reset()
    return env

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Train or test DQN model for highway-fast')
    argparser.add_argument('--env', type=str, help='environment ID', default='highway-v0')
    argparser.add_argument('--log_path', type=str, help='Logging Location', default='data/')
    argparser.add_argument('--batch_size', type=int, help='Batch size for training', default=32)
    args = argparser.parse_args()

    trainenvfunc = partial(train_env, args.env)
    testenvfunc = partial(test_env, args.env)

    log_path = os.path.join(args.log_path, "cnn_" + args.env.replace("-v0", "")+'/')

    # Train
    model = DQN('CnnPolicy', DummyVecEnv([trainenvfunc]),
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=200,
                batch_size=args.batch_size,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                exploration_fraction=0.7,
                verbose=2,
                tensorboard_log=log_path)
    
    env = DummyVecEnv([testenvfunc])
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
            
            return True
    
    eval_callback = EvalCallback(env, best_model_save_path=os.path.join(log_path, 'best_model'),
                             log_path=os.path.join(log_path, 'results'), eval_freq=500,
                             deterministic=True, render=False)
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_path)
    
    callback = CallbackList([checkpoint_callback, eval_callback, TensorboardCallback()])

    
    model.learn(total_timesteps=int(1e5), callback=callback)
    model.save(os.path.join(log_path, "model"))

    # Record video
    model = DQN.load(os.path.join(log_path, "model"))

    video_length = 2 * env.envs[0].config["duration"]
    env = VecVideoRecorder(
        env, 
        os.path.join(log_path, "videos"),
        record_video_trigger=lambda x: x == 0, 
        video_length=video_length,
        name_prefix="dqn-agent"
    )

    obs = env.reset()
    for _ in range(video_length + 1):
        action, _ = model.predict(obs)
        obs, _, _, _ = env.step(action)
    env.close()
