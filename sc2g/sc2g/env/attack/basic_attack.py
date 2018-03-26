#================================
# RESEARCH GROUP PROJECT [RGP]
#================================
# This file is part of the COMP3096 Research Group Project.

# System
import logging

# Gym Imports
import gym
from gym.spaces import Box, Discrete

# PySC2 Imports
from pysc2.env.sc2_env import SC2Env
from pysc2.lib.actions import FUNCTIONS, FunctionCall
from pysc2.lib.features import SCREEN_FEATURES

# Numpy
import numpy as np

# Typing
from typing import List

from sc2g.env.sc2gym import SC2GymEnv

# Setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AttackEnv(SC2GymEnv):
    def __init__(self, sc2_env: SC2Env, **kwargs):
        super().__init__(sc2_env, **kwargs)

        # Specify observation and action space
        screen_shape_observation = self.screen_shape + (1,)
        self.observation_space = Box(low=0, high=SCREEN_FEATURES.player_relative.scale, shape=screen_shape_observation)
        self.resolution = self.screen_shape[0] * self.screen_shape[1]
        self.action_space = Discrete(self.resolution * 2)  # width x height x (move/attack)
        self.move_space = self.resolution - 1 # unravel works from 0 to n-1
        self.unravel_shape = (self.screen_shape[0], self.screen_shape[1])

    def get_sc2_action(self, gym_action) -> List[FunctionCall]:
        # Get coords by unravelling action.
        # How unravel works:
        # Ref: https://www.quora.com/What-is-a-simple-intuitive-example-for-the-unravel_index-in-Python
        is_attack = gym_action > self.move_space
        if is_attack: gym_action -= self.resolution

        coords = np.unravel_index(gym_action, self.unravel_shape)

        # PySC2 uses different conventions for observations (y,x) and actions (x,y)
        # ::-1 reverses the tuple i.e. (1,2) becomes (2,1)
        action = FUNCTIONS.Attack_screen("now", coords[::-1]) if is_attack else FUNCTIONS.Move_screen("now", coords[::-1])

        if action.function not in self.available_actions:
            # logger.warning("Attempted unavailable action {}.".format(action))
            action = FUNCTIONS.select_army("select")

        return [action]
