import math
import warnings
import random
import magent
import numpy as np
from gym.spaces import Box, Discrete
from gym.utils import EzPickle

from pettingzoo import AECEnv
from pettingzoo.magent.render import Renderer
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import from_parallel_wrapper, parallel_wrapper_fn

from battle_v3 import KILL_REWARD, get_config
from magent_env import magent_parallel_env, make_env

default_map_size = 80
max_cycles_default = 1000
minimap_mode_default = False
default_reward_args = dict(step_reward=-0.005, dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2)


def parallel_env(map_size=default_map_size, max_cycles=max_cycles_default, minimap_mode=minimap_mode_default, extra_features=False, **reward_args):
    env_reward_args = dict(**default_reward_args)
    env_reward_args.update(reward_args)
    return _parallel_env(map_size, minimap_mode, env_reward_args, max_cycles, extra_features)


def raw_env(map_size=default_map_size, max_cycles=max_cycles_default, minimap_mode=minimap_mode_default, extra_features=False, **reward_args):
    return from_parallel_wrapper(parallel_env(map_size, max_cycles, minimap_mode, extra_features, **reward_args))


env = make_env(raw_env)


class _parallel_env(magent_parallel_env, EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array'], 'name': "battlefield_v3"}

    def __init__(self, map_size, minimap_mode, reward_args, max_cycles, extra_features):
        EzPickle.__init__(self, map_size, minimap_mode, reward_args, max_cycles, extra_features)
        assert map_size >= 46, "size of map must be at least 46"
        env = magent.GridWorld(get_config(map_size, minimap_mode, **reward_args), map_size=map_size)
        self.leftID = 0
        self.rightID = 1
        reward_vals = np.array([KILL_REWARD] + list(reward_args.values()))
        reward_range = [np.minimum(reward_vals, 0).sum(), np.maximum(reward_vals, 0).sum()]
        names = ["red", "blue"]
        super().__init__(env, env.get_handles(), names, map_size, max_cycles, reward_range, minimap_mode, extra_features)

    def generate_map(self):
        env, map_size, handles = self.env, self.map_size, self.handles
        """ generate a map, which consists of two squares of agents"""
        width = height = map_size
        init_num = map_size * map_size * 0.04
        gap = 3
        width = map_size
        height = map_size
        init_num = 20
        gap = 3
        leftID, rightID = 0, 1
        middle = map_size//2 # Based on size
        random.seed(10)
        # walls
        pos = []
        wall = 1 # 9 or 10
        self.noise=0.0
        off = 10
        for y in range(middle-wall-1+off,middle+wall+1+off):
            for x in range(middle-wall-1,middle+wall+1):
                if random.random() >= self.noise:
                    pos.append((x, y))
        for y in range(middle-wall-1-off,middle+wall+1-off):
            for x in range(middle-wall-1,middle+wall+1):
                if random.random() >= self.noise:
                    pos.append((x, y))
        env.add_walls(pos=pos, method="custom")
        
        # left
        pos = []
        for x in range(6, 9, 2):
            for y in range(middle-8*2,middle+8*2, 2):
                pos.append([x, y, 0])
        env.add_agents(handles[leftID], method="custom", pos=pos)
        
        # right
        pos = []
        for x in range(2*middle-9, 2*middle-6, 2):
            for y in range(middle-8*2,middle+8*2, 2):
                pos.append([x, y, 0])
        env.add_agents(handles[rightID], method="custom", pos=pos)
