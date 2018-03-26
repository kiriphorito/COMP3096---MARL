#================================
# RESEARCH GROUP PROJECT [RGP]
#================================
# This file is part of the COMP3096 Research Group Project.

# System
import logging

# Gym Imports
import gym
from gym.spaces import Box, Discrete, Tuple

# PySC2 Imports
from pysc2.lib.actions import FUNCTIONS, FunctionCall
from pysc2.lib.features import SCREEN_FEATURES

# Numpy
import numpy as np

# Typing
from typing import List

from sc2g.env.unit_tracking import UnitTrackingEnv

# Setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TupleEx(Tuple):
    def __init__(self, spaces):
        super().__init__(spaces)
        self.n = 1
        for i in range(0, len(spaces)):
            self.n *= spaces[i].n

class SelectedMovementEnv(UnitTrackingEnv):
    def __init__(self, sc2_env, **kwargs):
        super().__init__(sc2_env, **kwargs)

        # Number of marines (hardcoded)
        self.number_of_marines = 2
        # Specify observation and action space
        screen_shape_observation = self.screen_shape + (1,)
        self.observation_space = Box(low=0, high=SCREEN_FEATURES.player_relative.scale, shape=screen_shape_observation)
        self.resolution = self.screen_shape[0] * self.screen_shape[1]  # (width x height)
        self.action_space = Discrete(self.resolution * self.number_of_marines)
        self.unravel_shape = (self.screen_shape[0], self.screen_shape[1])

    def get_sc2_action(self, gym_action) -> List[FunctionCall]:
        # Get coords by unravelling action. DQN only supports returning an integer as action.
        # How unravel works:
        # Ref: https://www.quora.com/What-is-a-simple-intuitive-example-for-the-unravel_index-in-Python

        idx = int(gym_action / self.resolution)
        coords = gym_action % self.resolution
        coords = np.unravel_index(coords, self.unravel_shape)
        coords = (coords[0], coords[1])

        target_unit = self.state['player_units_stable'][idx]
        target_tag = target_unit.tag.item()
        player_unit_tags = [unit.tag.item() for unit in self.state["player_units"]]  # .item() to convert numpy.int64 to native python type (int)

        # PySC2 uses different conventions for observations (y,x) and actions (x,y)
        # ::-1 reverses the tuple i.e. (1,2) becomes (2,1)
        if target_tag not in player_unit_tags:
            actions = [FUNCTIONS.no_op()]
        else:
            actions = [FUNCTIONS.move_unit(target_tag, "now", coords[::-1])]

        return actions
