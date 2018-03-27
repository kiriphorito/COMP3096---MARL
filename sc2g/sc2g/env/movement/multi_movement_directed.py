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


# ==========================================================
# Only applies to movement-based mini-games with
# two friendly player units (eg. CollectMineralShards)
# ==========================================================
class MultiMovementDirectedEnv(UnitTrackingEnv):
    def __init__(self, sc2_env, **kwargs):
        super().__init__(sc2_env, **kwargs)

        # Number of marines and adjacency (hardcoded)
        self.number_of_marines = 2
        self.number_adjacency = 8
        # Specify observation and action space
        screen_shape_observation = self.screen_shape + (1,)
        self.observation_space = Box(low=0, high=SCREEN_FEATURES.player_relative.scale, shape=screen_shape_observation)
        self.resolution = self.screen_shape[0] * self.screen_shape[1]  # (width x height)
        self.action_space = Discrete(self.resolution)
        self.unravel_shape = (self.screen_shape[0], self.screen_shape[1])

    def get_sc2_action(self, gym_action) -> List[FunctionCall]:
        if len(self.state["player_units_stable"]) == 0:
            return [FUNCTIONS.no_op()]

        # Get coords by unravelling action. DQN only supports returning an integer as action.
        # How unravel works:
        # Ref: https://www.quora.com/What-is-a-simple-intuitive-example-for-the-unravel_index-in-Python
        coords = np.unravel_index(gym_action, self.unravel_shape)

        # Get gym action for each marine
        gym_action_1, gym_action_2 = (coords[0] % self.number_adjacency, coords[1] % self.number_adjacency)

        # Get current coordinates for each marine
        marine_1_stable = self.state["player_units_stable"][0]
        marine_2_stable = self.state["player_units_stable"][1]

        # Get tags for each marine
        marine_1_tag = marine_1_stable.tag.item()
        marine_2_tag = marine_2_stable.tag.item()

        # Get target coordinates for each marine
        marine_1_curr_xy = next((unit.x, unit.y) for unit in self.state["player_units"] if unit.tag.item() == marine_1_tag)
        marine_2_curr_xy = next((unit.x, unit.y) for unit in self.state["player_units"] if unit.tag.item() == marine_2_tag)

        def get_target_xy(num, curr_coords):
            #  0: Up
            #  1: Down
            #  2: Left
            #  3: Right
            #  4: Up + Left
            #  5: Up + Right
            #  6: Down + Left
            #  7: Down + Right
            target_xy = list(curr_coords)
            # Determine target position
            if num in (0, 4, 5):
                # Up
                target_xy[1] = max(0, curr_coords[1]-1)

            if num in (1, 6, 7):
                # Down
                target_xy[1] = min(self.screen_shape[1]-1, curr_coords[1]+1)

            if num in (2, 4, 6):
                # Left
                target_xy[0] = max(0, curr_coords[0]-1)

            if num in (3, 5, 7):
                # Right
                target_xy[0] = min(self.screen_shape[0]-1, curr_coords[0]+1)

            return tuple(target_xy)

        marine_1_target_xy = get_target_xy(gym_action_1, marine_1_curr_xy)
        marine_2_target_xy = get_target_xy(gym_action_2, marine_2_curr_xy)

        # Assign action functions
        actions = [FUNCTIONS.move_unit(marine_1_tag, "now", marine_1_target_xy), FUNCTIONS.move_unit(marine_2_tag, "now", marine_2_target_xy)]

        return actions
