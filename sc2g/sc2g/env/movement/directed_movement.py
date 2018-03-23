#================================
# RESEARCH GROUP PROJECT [RGP]
#================================
# This file is part of the COMP3096 Research Group Project.

# System
import logging

# Gym Imports
import gym
from gym.spaces import Discrete

# PySC2 Imports
from pysc2.env.sc2_env import SC2Env
from pysc2.env.environment import TimeStep
from pysc2.lib.actions import FUNCTIONS, FunctionCall
from pysc2.lib import features

# Typing
from typing import List

from sc2g.env.unit_tracking import UnitTrackingEnv

# Setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DirectedMovementEnv(UnitTrackingEnv):
    def __init__(self, sc2_env: SC2Env, **kwargs):
        super().__init__(sc2_env, **kwargs)
        self.action_space = Discrete(8)  # set to 4 for up, down, left, right, 8 for 8-way

    def get_sc2_action(self, gym_action) -> List[FunctionCall]:
        if FUNCTIONS.Move_screen.id not in self.available_actions:
            return[FUNCTIONS.select_army("select")]

        # player_xy = [(unit.x, unit.y) for unit in self.state["player_units"]][0]

        player_units_y, player_units_x = (self.state["player_relative"] == features.PlayerRelative.SELF).nonzero()

        if not player_units_x.size or not player_units_y.size:
            print("No coords: X: {} / Y: {}".format(player_units_x, player_units_y))
            return [FUNCTIONS.no_op()]

        player_units_xy = [int(player_units_x.mean()), int(player_units_y.mean())]
        target_xy = player_units_xy

        # 0: Up
        # 1: Down
        # 2: Left
        # 3: Right
        # 4: Up + left
        # 5: Up + right
        # 6: Down + left
        # 7: Down + right
        if gym_action in (0, 4, 5):
            # up
            target_xy[1] = 0
            pass
        if gym_action in (1, 6, 7):
            # down
            target_xy[1] = self.screen_shape[1] - 1
            pass
        if gym_action in (2, 4, 6):
            # left
            target_xy[0] = 0
            pass
        if gym_action in (3, 5, 7):
            # right
            target_xy[0] = self.screen_shape[0] - 1
            pass

        action = FUNCTIONS.Move_screen("now", target_xy)

        return [action]

    def update_state(self, timestep: TimeStep):
        super().update_state(timestep)
        self.state["player_relative"] = timestep.observation.feature_screen.player_relative
