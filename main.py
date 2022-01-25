from board import HexBoard, HexEnv
import tianshou as ts
from torch import nn
import torch
import numpy as np
from tianshou.utils.net.common import Net
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import gym
import numpy as np
import torch

from tianshou.data import (
    Batch,
    CachedReplayBuffer,
    ReplayBuffer,
    ReplayBufferManager,
    VectorReplayBuffer,
    to_numpy,
)
from tianshou.data.batch import _alloc_by_keys_diff
from tianshou.env import BaseVectorEnv, DummyVectorEnv
from tianshou.policy import BasePolicy

class Net(nn.Module):
    def conv(self, n_in, n_out):
        return nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=(3, 3))
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        
        self.model = nn.Sequential(
            self.conv(6, 8), nn.ReLU(inplace=True),
            self.conv(8, 16), nn.ReLU(inplace=True),
            self.conv(16, 64), nn.ReLU(inplace=True),
            nn.Flatten(), nn.Linear(64, 64), nn.ReLU(inplace=True),
            nn.Linear(64, np.prod(action_shape))
        )
    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        obs = obs.view((batch,) + self.state_shape)
        logits = self.model(obs)
        return logits, state

example_env = HexEnv(3, 2)
train_envs = ts.env.DummyVectorEnv([lambda: HexEnv(3, 2)])
test_envs = ts.env.DummyVectorEnv([lambda: HexEnv(3, 2)])
state_shape = example_env.observation_space.shape or example_env.observation_space.n
action_shape = example_env.action_space.shape or example_env.action_space.n
# print('shapes', state_shape, action_shape)
net = Net(state_shape, action_shape)
    
optim = torch.optim.Adam(net.parameters())
policy = ts.policy.DQNPolicy(net, optim, discount_factor=0.9, estimation_step=3, target_update_freq=320)
train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(2000, 10), exploration_noise=True)
test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

result = ts.trainer.offpolicy_trainer(
    policy, train_collector, test_collector,
    max_epoch=10, step_per_epoch=100, step_per_collect=10,
    update_per_step=0.1, episode_per_test=10, batch_size=64,
    train_fn=lambda epoch, env_step: policy.set_eps(0.1),
    test_fn=lambda epoch, env_step: policy.set_eps(0.05),
    stop_fn=lambda mean_rewards: mean_rewards >= 0)
print(f'Finished training! Use {result["duration"]}')


# # # get the next action
# # if random:
# #     self.data.update(
# #         act=[self._action_space[i].sample() for i in ready_env_ids]
# #     )
# # else:
# data = Batch(
#             obs={}, act={}, rew={}, done={}, obs_next={}, info={}, policy={}
#         )
# data.obs = train_envs.reset()
# spolicy = policy
# # print(data.obs)
# # print(len(data))
# last_state = data.policy.pop("hidden_state", None)
# result = spolicy(data, last_state)
# # update state / act / policy into self.data
# policy = result.get("policy", Batch())
# assert isinstance(policy, Batch)
# state = result.get("state", None)
# if state is not None:
#     policy.hidden_state = state  # save state into buffer
# # print('lol', result)
# act = to_numpy(result.act)
# data.update(policy=policy, act=act)

# # get bounded and remapped actions first (not saved into buffer)
# # data.act = np.ones((2, 2)) * 3
# action_remap = spolicy.map_action(data.act)
# # step in env
# print('kek', action_remap)
# result = train_envs.step(action_remap, np.arange(1))  # type: ignore
# # obs_next, rew, done, info = result