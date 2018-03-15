#================================
# RESEARCH GROUP PROJECT [RGP]
#================================
# This file is part of the COMP3096 Research Group Project.
#
# MultistepAgent is the base class for an agent that can perform multiple actions per timestep.
# A multi-action agent must implement the `multistep` function and return an array of actions to take.
#
# The step() function is *ignored* and any action taken there will be silently discarded.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.agents import base_agent
from pysc2.lib import actions

FUNCTIONS = actions.FUNCTIONS

class MultistepAgent(base_agent.BaseAgent):
  """Base class for an agent that can perform multiple actions per timestep.
  """

  # Return an array of actions.
  def multistep(self, obs):
    self.steps += 1
    self.reward += obs.reward
    return [FUNCTIONS.no_op()]
