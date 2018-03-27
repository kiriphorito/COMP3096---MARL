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

# Numpy
import numpy as np

# Typing
from typing import List

from sc2g.env.unit_tracking import UnitTrackingEnv

# Setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DirectedAttackEnv(UnitTrackingEnv):
    def __init__(self, sc2_env: SC2Env, **kwargs):
        super().__init__(sc2_env, **kwargs)
        self.adjacency = 8 # set to 4 for up, down, left, right, 8 for 8-way
        self.action_space = Discrete(self.adjacency * 2) # Move/Attack
        self.move_space = (self.action_space.n / 2) - 1

    def get_sc2_action(self, gym_action) -> List[FunctionCall]:
        if FUNCTIONS.Move_screen.id not in self.available_actions:
            return [FUNCTIONS.select_army("select")]

        # For restricting the action space for A2C, does not break DQN
        gym_action %= self.action_space.n

        # Find mean coordinates of all currently active player units
        player_units_xy = [(unit.x, unit.y) for unit in self.state["player_units"]]
        arr = np.asarray(player_units_xy)
        length = arr.shape[0]
        x_sum = np.sum(arr[:, 0])
        y_sum = np.sum(arr[:, 1])
        centroid = (int(x_sum/length), int(y_sum/length))

        #  0: Up
        #  1: Down
        #  2: Left
        #  3: Right
        #  4: Up + Left
        #  5: Up + Right
        #  6: Down + Left
        #  7: Down + Right
        #  8: Up + Attack
        #  9: Down + Attack
        # 10: Left + Attack
        # 11: Right + Attack
        # 12: Up + Left + Attack
        # 13: Up + Right + Attack
        # 14: Down + Left + Attack
        # 15: Down + Right + Attack

        is_attack = gym_action > self.move_space
        if is_attack: gym_action %= self.adjacency
        target_xy = list(centroid)
        x_max = self.screen_shape[0]-1
        y_max = self.screen_shape[1]-1
        # Determine target position, diff => min(abs(x_diff), abs(y_diff))
        if   gym_action == 0: target_xy[1] = 0
        elif gym_action == 1: target_xy[1] = y_max
        elif gym_action == 2: target_xy[0] = 0
        elif gym_action == 3: target_xy[0] = x_max
        elif gym_action == 4:
            diff = min(centroid[0], centroid[1])
            target_xy = [target_xy[0] - diff, target_xy[1] - diff]
        elif gym_action == 5:
            diff = min(x_max - centroid[0], centroid[1])
            target_xy = [target_xy[0] + diff, target_xy[1] - diff]
        elif gym_action == 6:
            diff = min(centroid[0], y_max - centroid[1])
            target_xy = [target_xy[0] - diff, target_xy[1] + diff]
        elif gym_action == 7:
            diff = min(x_max - centroid[0], y_max - centroid[1])
            target_xy = [target_xy[0] + diff, target_xy[1] + diff]

        # Assign action function
        action = FUNCTIONS.Attack_screen("now", target_xy) if is_attack else FUNCTIONS.Move_screen("now", target_xy)

        return [action]
