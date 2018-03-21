#================================
# RESEARCH GROUP PROJECT [RGP]
#================================
# This file is part of the COMP3096 Research Group Project.
# Note: This file is modified from env/movement_minigame.py.
# Changes are marked with `RGP`.
#
# Important notes:
# - The action space of this environment is spaces.Discrete(2) to support two agents only.

import numpy as np
from gym import spaces
from pysc2.lib import actions, features

from sc2gym.envs import SC2GameEnv

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_RELATIVE_SCALE = features.SCREEN_FEATURES.player_relative.scale

_NO_OP = actions.FUNCTIONS.no_op.id

_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_NOT_QUEUED = [0]

# RGP
FUNCTIONS = actions.FUNCTIONS
_MOVE_MARINE = FUNCTIONS.Move_marine.id

# Unit-based movement.
class BaseUnitMovementEnv(SC2GameEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._action_space = None
        self._observation_space = None

    def _reset(self):
        super()._reset()
        return self._post_reset()

    def _post_reset(self):
        # First thing to do post-reset - select the entire army.
        obs, reward, done, info = self._safe_step([_SELECT_ARMY, _SELECT_ALL])
        obs = self._extract_observation(obs)
        return obs

    def _step(self, action):
        action = self._translate_action(action)
        obs, reward, done, info = self._safe_step(action)
        if obs is None:
            return None, 0, True, {}
        obs = self._extract_observation(obs)
        return obs, reward, done, info

    @property
    def observation_space(self):
        if self._observation_space is None:
            self._observation_space = self._get_observation_space()
        return self._observation_space

    def _get_observation_space(self):
        screen_shape = (1, ) + self.observation_spec[0]["feature_screen"][1:]
        space = spaces.Box(low=0, high=_PLAYER_RELATIVE_SCALE, shape=screen_shape)
        return space

    @property
    def action_space(self):
        if self._action_space is None:
            self._action_space = self._get_action_space()
        return self._action_space

    def _get_action_space(self):
        return spaces.Discrete(2)

    def _extract_observation(self, obs):
        obs = obs.observation["feature_screen"][_PLAYER_RELATIVE]
        obs = obs.reshape(self.observation_space.shape)
        return obs

    def _translate_action(self, action):
        if action < 0 or action > self.action_space.n:
            print('error: action out of range')
            return [_NO_OP]

        # note: action = index of marine to move
        # TODO: convert index to tag or vice versa
        return [_MOVE_MARINE, _NOT_QUEUED, action]
