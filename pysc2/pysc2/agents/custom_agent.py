from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
FUNCTIONS = actions.FUNCTIONS


class CustomAgent(base_agent.BaseAgent):
  """Independent two-agent control for the CollectMineralShards minigame."""

  def setup(self, obs_spec, action_spec):
    super(CustomAgent, self).setup(obs_spec, action_spec)
    if "feature_units" not in obs_spec:
      raise Exception("This agent requires the feature_units observation.")

  def reset(self):
    super(CustomAgent, self).reset()
    self._current_marine_tag = 0
    self._previous_mineral_xy = [-1, -1]

  def step(self, obs):
    super(CustomAgent, self).step(obs)

    # Get list of marines
    marines = [unit for unit in obs.observation.feature_units
               if unit.alliance == _PLAYER_SELF]
    if not marines:
      return FUNCTIONS.no_op()

    # Select other marine unit
    marine_unit = next(marine for marine in marines if marine.tag != self._current_marine_tag)
    self._current_marine_tag = marine_unit.tag
    marine_xy = [marine_unit.x, marine_unit.y]

    # Get list of mineral locations
    minerals = [[unit.x, unit.y] for unit in obs.observation.feature_units
                if unit.alliance == _PLAYER_NEUTRAL]

    if self._previous_mineral_xy in minerals:
      # Remove the previous target of the other marine from consideration
      minerals.remove(self._previous_mineral_xy)

    if minerals:
      # Find the closest mineral.
      distances = numpy.linalg.norm(
        numpy.array(minerals) - numpy.array(marine_xy), axis=1)
      closest_mineral_xy = minerals[numpy.argmin(distances)]
      self._previous_mineral_xy = closest_mineral_xy  # Record which mineral we selected
      return FUNCTIONS.move_unit(marine_unit.tag.item(), "now", closest_mineral_xy) # .item() to convert numpy.int64 to native python type (int)

    return FUNCTIONS.no_op()

  def multistep(self, obs):
    return []
