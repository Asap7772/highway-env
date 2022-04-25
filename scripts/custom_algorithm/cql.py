import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from .offline_algorithm import OfflineAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, is_vectorized_observation, polyak_update
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy
import torch
import wandb
import math


class CQL(OfflineAlgorithm):
    """
    Deep Q-Network (DQN)

    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Default hyperparameters are taken from the nature paper,
    except for the optimizer and learning rate that were taken from Stable Baselines defaults.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 0,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        with_minq=False,
        min_q_weight=1.0,
        method=False,
        method_temp=50.0,
        wandb_log=False,
        wandb_log_freq=100,
    ):

        super(CQL, self).__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Discrete,),
            support_multi_env=True,

        )

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        # For updating the target network with multiple envs:
        self._n_calls = 0
        self.max_grad_norm = max_grad_norm
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0
        # Linear schedule will be defined in `_setup_model()`
        self.exploration_schedule = None
        self.q_net, self.q_net_target = None, None

        self.wandb_log = wandb_log
        self.wandb_log_freq = wandb_log_freq

        self.with_minq = with_minq
        self.min_q_weight = min_q_weight
        self.method = method
        self.method_temp = method_temp


        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(CQL, self)._setup_model()
        self._create_aliases()
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )
        # Account for multiple environments
        # each call to step() corresponds to n_envs transitions
        if self.n_envs > 1:
            if self.n_envs > self.target_update_interval:
                warnings.warn(
                    "The number of environments used is greater than the target network "
                    f"update interval ({self.n_envs} > {self.target_update_interval}), "
                    "therefore the target network will be updated after each call to env.step() "
                    f"which corresponds to {self.n_envs} steps."
                )

            self.target_update_interval = max(self.target_update_interval // self.n_envs, 1)

    def _create_aliases(self) -> None:
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self._n_calls += 1
        if self._n_calls % self.target_update_interval == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

    def get_qval(self, obs: th.Tensor, action: th.Tensor) -> th.Tensor:
        """
        Get the Q-value for the given observation and action.
        """
        qvals_all = self.q_net(obs)
        return th.gather(qvals_all, dim=1, index=action.long())

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            q1_vals_all = current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            q_data = current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            td_loss = loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')


            if self.with_minq:
                orig_push_up = push_up = self.get_qval(replay_data.observations, replay_data.actions)

                if self.method:
                    temp = math.exp(self.method_temp)
                    norm_const = torch.log(torch.tensor(q1_vals_all.shape[-1]))
                    q_new = q_data/(temp + 1e-9) - torch.logsumexp(q1_vals_all/temp, dim=-1).unsqueeze(-1)
                    weight = torch.exp(q_new).detach()/norm_const
                    push_up = orig_push_up * weight

                current_q_values = self.q_net(replay_data.observations)
                push_down = torch.logsumexp(current_q_values, 1).unsqueeze(-1)
                min_qf1_loss = self.min_q_weight * (push_down-push_up)
                orig_min_qf1_loss = self.min_q_weight * (push_down-orig_push_up)
                loss = loss + min_qf1_loss
                
            full_loss = loss.mean().item()
            losses.append(full_loss)
            
            loss = loss.mean()

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
                

        # Increase update counter
        self._n_updates += gradient_steps

        """
        Log the loss and the mean and std of the Q-values
        """
        things_to_log = {
            'train/n_updates': self._n_updates,
            'train/loss': full_loss,
            
            'train/data_q_mean': np.mean(q_data.detach().cpu().numpy()),
            'train/data_q_std': np.std(q_data.detach().cpu().numpy()),
            'train/data_q_min': np.min(q_data.detach().cpu().numpy()),
            'train/data_q_max': np.max(q_data.detach().cpu().numpy()),

            'train/all_q_mean': np.mean(q1_vals_all.detach().cpu().numpy()),
            'train/all_q_std': np.std(q1_vals_all.detach().cpu().numpy()),
            'train/all_q_min': np.min(q1_vals_all.detach().cpu().numpy()),
            'train/all_q_max': np.max(q1_vals_all.detach().cpu().numpy()),

            'train/cql_loss_mean': np.mean(min_qf1_loss.detach().cpu().numpy()),
            'train/cql_loss_std': np.std(min_qf1_loss.detach().cpu().numpy()),
            'train/cql_loss_max': np.max(min_qf1_loss.detach().cpu().numpy()),
            'train/cql_loss_min': np.min(min_qf1_loss.detach().cpu().numpy()),

            'train/orig_cql_loss_mean': np.mean(orig_min_qf1_loss.detach().cpu().numpy()),
            'train/orig_cql_loss_std': np.std(orig_min_qf1_loss.detach().cpu().numpy()),
            'train/orig_cql_loss_max': np.max(orig_min_qf1_loss.detach().cpu().numpy()),
            'train/orig_cql_loss_min': np.min(orig_min_qf1_loss.detach().cpu().numpy()),
            
            'train/td_loss_mean': np.mean(td_loss.detach().cpu().numpy()),
            'train/td_loss_std': np.std(td_loss.detach().cpu().numpy()),
            'train/td_loss_max': np.max(td_loss.detach().cpu().numpy()),
            'train/td_loss_min': np.min(td_loss.detach().cpu().numpy()),
        }

        if self.method:
            method_dict = {
                'train/method_temp': self.method_temp,
                'train/weight_mean': np.mean(weight.detach().cpu().numpy()),
                'train/weight_std': np.std(weight.detach().cpu().numpy()),
                'train/weight_min': np.min(weight.detach().cpu().numpy()),
                'train/weight_max': np.max(weight.detach().cpu().numpy()),
            }
            things_to_log.update(method_dict)

        for key, value in things_to_log.items():
            self.logger.record(key, value)

        if self.wandb_log:
            if self._n_updates % self.wandb_log_freq == 0:
                wandb.log(things_to_log)

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            if is_vectorized_observation(maybe_transpose(observation, self.observation_space), self.observation_space):
                if isinstance(self.observation_space, gym.spaces.Dict):
                    n_batch = observation[list(observation.keys())[0]].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic)
        return action, state

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "CQL",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True):

        return super(CQL, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def _excluded_save_params(self) -> List[str]:
        return super(CQL, self)._excluded_save_params() + ["q_net", "q_net_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []