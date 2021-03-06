#!/usr/bin/python

#================================
# RESEARCH GROUP PROJECT [RGP]
#================================
# This file is part of the COMP3096 Research Group Project.
# Note: This file is modified from bin/agent.py and uses SC2MarlEnv instead of SC2Env.
#
# Changes:
# - Use pysc2.env.sc2_marl_env.SC2MarlEnv in place of pysc2.env.sc2_env.SC2Env.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import threading

from absl import app
from absl import flags
from future.builtins import range  # pylint: disable=redefined-builtin

from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import run_loop
from pysc2.env import sc2_marl_env
from pysc2.lib import stopwatch


FLAGS = flags.FLAGS
flags.DEFINE_bool("render", True, "Whether to render with pygame.")
flags.DEFINE_integer("feature_screen_size", 84,
                     "Resolution for screen feature layers.")
flags.DEFINE_integer("feature_minimap_size", 64,
                     "Resolution for minimap feature layers.")
flags.DEFINE_integer("rgb_screen_size", None,
                     "Resolution for rendered screen.")
flags.DEFINE_integer("rgb_minimap_size", None,
                     "Resolution for rendered minimap.")
flags.DEFINE_enum("action_space", None, sc2_marl_env.ActionSpace._member_names_,  # pylint: disable=protected-access
                  "Which action space to use. Needed if you take both feature "
                  "and rgb observations.")
flags.DEFINE_bool("use_feature_units", False,
                  "Whether to include feature units.")

flags.DEFINE_integer("max_agent_steps", 0, "Total agent steps.")
flags.DEFINE_integer("game_steps_per_episode", 0, "Game steps per episode.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

flags.DEFINE_string("agent", "pysc2.agents.random_agent.RandomAgent",
                    "Which agent to run, as a python path to an Agent class.")
flags.DEFINE_enum("agent_race", "random", sc2_marl_env.Race._member_names_,  # pylint: disable=protected-access
                  "Agent 1's race.")

flags.DEFINE_string("agent2", "Bot", "Second agent, either Bot or agent class.")
flags.DEFINE_enum("agent2_race", "random", sc2_marl_env.Race._member_names_,  # pylint: disable=protected-access
                  "Agent 2's race.")
flags.DEFINE_enum("difficulty", "very_easy", sc2_marl_env.Difficulty._member_names_,  # pylint: disable=protected-access
                  "If agent2 is a built-in Bot, it's strength.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")

flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")

flags.DEFINE_string("map", None, "Name of a map to use.")
flags.mark_flag_as_required("map")


def run_thread(agent_classes, players, map_name, visualize):
  with sc2_marl_env.SC2MarlEnv(
      map_name=map_name,
      players=players,
      step_mul=FLAGS.step_mul,
      game_steps_per_episode=FLAGS.game_steps_per_episode,
      feature_screen_size=FLAGS.feature_screen_size,
      feature_minimap_size=FLAGS.feature_minimap_size,
      rgb_screen_size=FLAGS.rgb_screen_size,
      rgb_minimap_size=FLAGS.rgb_minimap_size,
      action_space=(FLAGS.action_space and
                    sc2_marl_env.ActionSpace[FLAGS.action_space]),
      use_feature_units=FLAGS.use_feature_units,
      visualize=visualize) as env:
    env = available_actions_printer.AvailableActionsPrinter(env)
    agents = [agent_cls() for agent_cls in agent_classes]
    run_loop.run_loop(agents, env, FLAGS.max_agent_steps)
    if FLAGS.save_replay:
      env.save_replay(agent_classes[0].__name__)


def main(unused_argv):
  """Run an agent."""
  stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
  stopwatch.sw.trace = FLAGS.trace

  maps.get(FLAGS.map)  # Assert the map exists.

  agent_classes = []
  players = []

  agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
  agent_cls = getattr(importlib.import_module(agent_module), agent_name)
  agent_classes.append(agent_cls)
  players.append(sc2_marl_env.Agent(sc2_marl_env.Race[FLAGS.agent_race]))

  if FLAGS.agent2 == "Bot":
    players.append(sc2_marl_env.Bot(sc2_marl_env.Race[FLAGS.agent2_race],
                               sc2_marl_env.Difficulty[FLAGS.difficulty]))
  else:
    agent_module, agent_name = FLAGS.agent2.rsplit(".", 1)
    agent_cls = getattr(importlib.import_module(agent_module), agent_name)
    agent_classes.append(agent_cls)
    players.append(sc2_marl_env.Agent(sc2_marl_env.Race[FLAGS.agent2_race]))

  threads = []
  for _ in range(FLAGS.parallel - 1):
    t = threading.Thread(target=run_thread,
                         args=(agent_classes, players, FLAGS.map, False))
    threads.append(t)
    t.start()

  run_thread(agent_classes, players, FLAGS.map, FLAGS.render)

  for t in threads:
    t.join()

  if FLAGS.profile:
    print(stopwatch.sw)


def entry_point():  # Needed so setup.py scripts work.
  app.run(main)


if __name__ == "__main__":
  app.run(main)
