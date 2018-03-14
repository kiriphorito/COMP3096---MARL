#================================
# RESEARCH GROUP PROJECT [RGP]
#================================
# This modified file is part of the COMP3096 Research Group Project.
# Changes are marked with `RGP`.
#
# Summary of changes:
# - For agents inheriting from the MultistepAgent base class, the `multistep` function is called to obtain the array of
# actions that the agent would like to take.
# - The environment is stepped forward with both 'standard actions' (single actions returned from non multi-step agents)
# and multistep actions.

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A run loop for agent/environment interaction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

# RGP
from pysc2.agents import multistep_agent

def run_loop(agents, env, max_frames=0):
  """A run loop to have agents and an environment interact."""
  total_frames = 0
  start_time = time.time()

  observation_spec = env.observation_spec()
  action_spec = env.action_spec()
  for agent, obs_spec, act_spec in zip(agents, observation_spec, action_spec):
    agent.setup(obs_spec, act_spec)

  try:
    while True:
      timesteps = env.reset()
      for a in agents:
        a.reset()
      while True:
        total_frames += 1

        # RGP {
        standard_agents = [agent for agent in agents if not isinstance(agent, multistep_agent.MultistepAgent)]
        multiagent_agents = [agent for agent in agents if isinstance(agent, multistep_agent.MultistepAgent)]

        # Standard actions
        actions = [agent.step(timestep)
                   for agent, timestep in zip(standard_agents, timesteps)]

        # Multistep actions
        multiagent_actions_list = [agent.multistep(timestep) for agent, timestep in zip(multiagent_agents, timesteps)] # list of lists
        multiagent_actions = [action for sublist in multiagent_actions_list for action in sublist] # flatten the list to get a list of actions

        # } RGP

        if max_frames and total_frames >= max_frames:
          return
        if timesteps[0].last():
          break
        timesteps = env.step(actions + multiagent_actions) # RGP: (+ multiagent_actions)
  except KeyboardInterrupt:
    pass
  finally:
    elapsed_time = time.time() - start_time
    print("Took %.3f seconds for %s steps: %.3f fps" % (
        elapsed_time, total_frames, total_frames / elapsed_time))
