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

# ==========================================================
# Only applies to single player unit game (eg. MoveToBeacon)
# ==========================================================
class SingleDirectedMovementEnv(UnitTrackingEnv):
    def __init__(self, sc2_env: SC2Env, **kwargs):
        super().__init__(sc2_env, **kwargs)
        self.action_space = Discrete(9)  # set to 4 for up, down, left, right, 8 for 8-way, 9th is no-op

    def get_sc2_action(self, gym_action) -> List[FunctionCall]:
        if FUNCTIONS.Move_screen.id not in self.available_actions:
            return[FUNCTIONS.select_army("select")]

        # 0 = no-op
        if gym_action == 0:
            return [FUNCTIONS.no_op()]

        player_unit_xy = self.state["player_units"][0]
        target_xy = player_unit_xy

        #  0: No-op
        #  1: Up
        #  2: Down
        #  3: Left
        #  4: Right
        #  5: Up + Left
        #  6: Up + Right
        #  7: Down + Left
        #  8: Down + Right

        # Determine target position
        if gym_action in (1, 5, 6):
            # Up
            target_xy[1] = max(0, player_unit_xy[1]-1)

        if gym_action in (2, 7, 8):
            # Down
            target_xy[1] = min(self.screen_shape[1]-1, player_unit_xy[1]+1)

        if gym_action in (3, 5, 7):
            # Left
            target_xy[0] = max(0, player_unit_xy[0]-1)

        if gym_action in (4, 6, 8):
            # Right
            target_xy[0] = min(self.screen_shape[0]-1, player_unit_xy[0]+1)

        # Assign action function
        # Move_screen
        action = FUNCTIONS.Move_screen("now", target_xy)

        return [action]

    def update_state(self, timestep: TimeStep, is_new_episode):
        super().update_state(timestep, is_new_episode)
        self.state["player_relative"] = timestep.observation.feature_screen.player_relative
