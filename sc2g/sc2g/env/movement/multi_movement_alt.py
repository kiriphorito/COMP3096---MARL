#================================
# RESEARCH GROUP PROJECT [RGP]
#================================
# This file is part of the COMP3096 Research Group Project.

# Gym Imports
import gym
from gym.spaces import Box, Discrete, Tuple

# PySC2 Imports
from pysc2.lib.actions import FUNCTIONS, FunctionCall
from pysc2.lib.features import SCREEN_FEATURES
from pysc2.env.environment import TimeStep

# Numpy
import numpy as np

# Typing
from typing import List

from sc2g.env.unit_tracking import UnitTrackingEnv


class MultiMovementAltEnv(UnitTrackingEnv):
    def __init__(self, sc2_env, **kwargs):
        super().__init__(sc2_env, **kwargs)

        self.action_space = Discrete(self.screen_shape[0] * self.screen_shape[1])  # width x height

        self.state["unit_to_move"] = 0

    def get_sc2_action(self, gym_action) -> List[FunctionCall]:
        # Get coords by unravelling action.
        # How unravel works:
        # Ref: https://www.quora.com/What-is-a-simple-intuitive-example-for-the-unravel_index-in-Python
        coords = np.unravel_index(gym_action, (self.screen_shape[0], self.screen_shape[1]))

        # PySC2 uses different conventions for observations (y,x) and actions (x,y)
        # ::-1 reverses the tuple i.e. (1,2) becomes (2,1)
        if self.state['player_unit_stable_tags']:
            tag_to_move = self.state['player_unit_stable_tags'][self.state["unit_to_move"]]
            actions = [FUNCTIONS.move_unit(tag_to_move, "now", coords[::-1])]
        else:
            actions = [FUNCTIONS.no_op()]

        return actions

    def update_state(self, timestep: TimeStep, is_new_episode):
        super().update_state(timestep, is_new_episode)
        if (is_new_episode):
            self.state["unit_to_move"] = 0
        else:
            self.state["unit_to_move"] = 1 - self.state["unit_to_move"]