#================================
# RESEARCH GROUP PROJECT [RGP]
#================================
# This file is part of the COMP3096 Research Group Project.

from sc2g.env.sc2gym import SC2GymEnv

# PySC2 Imports
from pysc2.env.environment import TimeStep
from pysc2.lib import features

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
_PLAYER_ENEMY = features.PlayerRelative.ENEMY


class UnitTrackingEnv(SC2GymEnv):
    def __init__(self, sc2_env, **kwargs):
        super().__init__(sc2_env, **kwargs)

        self.state['player_units'] = []
        self.state['neutral_units'] = []

    def update_state(self, timestep: TimeStep):
        # Get list of player units
        self.state['player_units'] = [unit for unit in timestep.observation.feature_units
                                      if unit.alliance == _PLAYER_SELF]

        # Get list of neutral units
        self.state['neutral_units'] = [unit for unit in timestep.observation.feature_units
                                       if unit.alliance == _PLAYER_NEUTRAL]

        # Get list of enemy units
        self.state['enemy_units'] = [unit for unit in timestep.observation.feature_units
                                     if unit.alliance == _PLAYER_ENEMY]

        # Sort units by tag
        self.state['player_units'].sort(key=lambda unit: unit.tag)
        self.state['neutral_units'].sort(key=lambda unit: unit.tag)
        self.state['enemy_units'].sort(key=lambda unit: unit.tag)
