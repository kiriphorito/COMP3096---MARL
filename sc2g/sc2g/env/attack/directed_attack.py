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


class DirectedAttackEnv(UnitTrackingEnv):
    def __init__(self, sc2_env: SC2Env, **kwargs):
        super().__init__(sc2_env, **kwargs)
        # 0th is no-op
        # 1st - 8th are Move_screen
        # 9th - 16th are Attack_screen
        self.action_space = Discrete(17)

    def get_sc2_action(self, gym_action) -> List[FunctionCall]:
        if FUNCTIONS.Move_screen.id not in self.available_actions:
            return [FUNCTIONS.select_army("select")]

        player_units_y, player_units_x = (self.state["player_relative"] == features.PlayerRelative.SELF).nonzero()

        if not player_units_x.size or not player_units_y.size:
            print("No coords: X: {} / Y: {}".format(player_units_x, player_units_y))
            return [FUNCTIONS.no_op()]

        player_units_xy = [int(player_units_x.mean()), int(player_units_y.mean())]
        target_xy = player_units_xy

        #  0: No-op
        #  1: Up
        #  2: Down
        #  3: Left
        #  4: Right
        #  5: Up + Left
        #  6: Up + Right
        #  7: Down + Left
        #  8: Down + Right
        #  9: Up + Attack
        # 10: Down + Attack
        # 11: Left + Attack
        # 12: Right + Attack
        # 13: Up + Left + Attack
        # 14: Up + Right + Attack
        # 15: Down + Left + Attack
        # 16: Down + Right + Attack

        # Determine target position
        if gym_action in (1, 5, 6, 9, 13, 14):
            # Up
            # target_xy[1] = min(player_units_xy[1]+1, self.screen_shape[1]-1)
            target_xy[1] = self.screen_shape[1]-1

        if gym_action in (2, 7, 8, 10, 15, 16):
            # Down
            # target_xy[1] = max(player_units_xy[1]-1, 0)
            target_xy[1] = 0

        if gym_action in (3, 5, 7, 11, 13, 15):
            # Left
            # target_xy[0] = max(player_units_xy[0]-1, 0)
            target_xy[0] = 0

        if gym_action in (4, 6, 8, 12, 14, 16):
            # Right
            # target_xy[0] = min(player_units_xy[0]+1, self.screen_shape[0]-1)
            target_xy[0] = self.screen_shape[0]-1

        # Assign action function
        if gym_action in range(1, 9):
            # Move_screen
            action = FUNCTIONS.Move_screen("now", target_xy)
        elif gym_action in range(9, 17):
            # Attack_screen
            action = FUNCTIONS.Attack_screen("now", target_xy)
        else:
            # No-op
            action = FUNCTIONS.no_op()

        return [action]

    def update_state(self, timestep: TimeStep):
        super().update_state(timestep)
        self.state["player_relative"] = timestep.observation.feature_screen.player_relative
