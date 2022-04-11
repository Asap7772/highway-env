import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

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
    argparser.add_argument('--log_path', type=str, help='Logging Location', default='')
    args = argparser.parse_args()

    trainenvfunc = partial(train_env, args.env)
    testenvfunc = partial(test_env, args.env)

    log_path = os.path.join(args.log_path, "highway_cnn_" + args.env.replace("-v0", "")+'/')

    # Train
    model = DQN('CnnPolicy', DummyVecEnv([trainenvfunc]),
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=200,
                batch_size=32,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                exploration_fraction=0.7,
                verbose=1,
                tensorboard_log=log_path)
    
    model.learn(total_timesteps=int(1e5))
    model.save(os.path.join(log_path, "model"))

    # Record video
    model = DQN.load(os.path.join(log_path, "model"))

    env = DummyVecEnv([testenvfunc])
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
