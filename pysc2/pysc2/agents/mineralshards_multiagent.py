#================================
# RESEARCH GROUP PROJECT [RGP]
#================================
# This file is part of the COMP3096 Research Group Project.
#
# MineralShardsMultiAgent is a scripted agent that controls two marines independently and
# performs multiple actions per timestep (one action for each marine).

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import time

from pysc2.agents import multistep_agent
from pysc2.lib import actions
from pysc2.lib import features

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
FUNCTIONS = actions.FUNCTIONS


class MineralShardsMultiAgent(multistep_agent.MultistepAgent):
  """Scripted, independent two-agent control for the CollectMineralShards minigame."""

  def setup(self, obs_spec, action_spec):
    super(MineralShardsMultiAgent, self).setup(obs_spec, action_spec)
    if "feature_units" not in obs_spec:
      raise Exception("This agent requires the feature_units observation.")

  def reset(self):
    super(MineralShardsMultiAgent, self).reset()
    self._marine_targets = {} # dictionary of current marine targets, with the marine's tag (unique ID) as the key and target_xy as value.

  # Return an array of actions to take.
  def multistep(self, obs):
    super(MineralShardsMultiAgent, self).step(obs)

    # Uncomment this to see the agent taking actions step-by-step.
    # time.sleep(0.8)

    # Get list of marines
    marines = [unit for unit in obs.observation.feature_units
               if unit.alliance == _PLAYER_SELF]

    # Bail if no marines
    if not marines:
      return [FUNCTIONS.no_op()]

    # Get list of mineral locations
    minerals = [[unit.x, unit.y] for unit in obs.observation.feature_units
                if unit.alliance == _PLAYER_NEUTRAL]

    # Bail if no minerals
    if not minerals:
      return [FUNCTIONS.no_op()]

    # Loop through marines
    actions = []
    for marine in marines:
      marine_xy = [marine.x, marine.y]

      # Remove the previous target of the other marine from consideration
      other_targets = [target_xy for (tag, target_xy) in self._marine_targets.items() if tag != marine.tag]
      other_target_xy = other_targets[0] if other_targets else (-1, -1)
      minerals_noprevious = [x for x in minerals if x != other_target_xy] if len(minerals) > 1 else minerals

      # Find the closest mineral.
      distances = numpy.linalg.norm(numpy.array(minerals_noprevious) - numpy.array(marine_xy), axis=1)
      closest_mineral_xy = minerals_noprevious[numpy.argmin(distances)]

      # Update the target of this marine
      self._marine_targets[marine.tag.item()] = closest_mineral_xy

      # Make the action
      action = FUNCTIONS.move_unit(marine.tag.item(), "now", closest_mineral_xy) # .item() to convert numpy.int64 to native python type (int)
      actions.append(action)

    return actions if actions else [FUNCTIONS.no_op()]
