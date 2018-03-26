#================================
# RESEARCH GROUP PROJECT [RGP]
#================================
# This file is part of the COMP3096 Research Group Project.
# This environment moves only the marine specified in the environment constructor.
# For use with multi-agent parallel environments.

# Gym Imports
from gym.spaces import Discrete

# PySC2 Imports
from pysc2.lib.actions import FUNCTIONS, FunctionCall

# Numpy
import numpy as np

# Typing
from typing import List

from sc2g.env.unit_tracking import UnitTrackingEnv


class MultiMovementIndexedEnv(UnitTrackingEnv):
    def __init__(self, sc2_env, unit_to_move, **kwargs):
        super().__init__(sc2_env, **kwargs)

        self.unit_to_move = unit_to_move
        self.action_space = Discrete(self.screen_shape[0] * self.screen_shape[1])  # width x height

    def get_sc2_action(self, gym_action) -> List[FunctionCall]:

        # Get coords by unravelling action.
        # How unravel works:
        # Ref: https://www.quora.com/What-is-a-simple-intuitive-example-for-the-unravel_index-in-Python
        coords = np.unravel_index(gym_action, (self.screen_shape[0], self.screen_shape[1]))

        # PySC2 uses different conventions for observations (y,x) and actions (x,y)
        # ::-1 reverses the tuple i.e. (1,2) becomes (2,1)
        if self.state['player_unit_stable_tags']:
            tag_to_move = self.state['player_unit_stable_tags'][self.unit_to_move]
            actions = [FUNCTIONS.move_unit(tag_to_move, "now", coords[::-1])]
        else:
            actions = [FUNCTIONS.no_op()]

        return actions
