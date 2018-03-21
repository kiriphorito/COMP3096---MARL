# System
import logging

# Gym Imports
import gym
from gym.spaces import Box, Discrete

# PySC2 Imports
from pysc2.env.sc2_env import SC2Env
from pysc2.env.environment import TimeStep
from pysc2.lib.actions import FUNCTIONS
from pysc2.lib.features import SCREEN_FEATURES

# Numpy
import numpy as np

# Typing
from typing import Tuple, Dict, Any

# Setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def make_env(map_name, id=0, **kwargs):
    default_args = dict(
        map_name=map_name,
        feature_screen_size=84,
        feature_minimap_size=64,
    )
    args = {**default_args, **kwargs}
    env = SC2Env(**args)
    return SC2GymEnv(env, id=id)


class SC2GymEnv(gym.Env):
    def __init__(self, sc2_env: SC2Env,
                 id=0,
                 print_freq=1,
                 agg_n_episodes=100):

        # Store settings/args
        self.sc2_env = sc2_env
        self.id = id
        self.print_freq = print_freq
        self.agg_n_episodes = agg_n_episodes

        # Get observation and action spaces from the SC2 environment
        self.observation_spec = self.sc2_env.observation_spec()
        self.screen_shape = self.observation_spec[0]["feature_screen"][1:]
        screen_shape_observation = self.screen_shape + (1,)  # RGP: Switched this around to put 1 at the back - this
        # is what tensorflow expects

        # Convert to Gym spaces and set observation and action space
        self.observation_space = Box(low=0, high=SCREEN_FEATURES.player_relative.scale, shape=screen_shape_observation)
        self.action_space = Discrete(self.screen_shape[0] * self.screen_shape[1])  # width x height

        # Status tracking
        self.elapsed_steps = 0
        self.elapsed_episodes = 0
        self.rolling_episode_score = np.zeros(agg_n_episodes, dtype=np.float32)

    # Called whenever a new episode starts.
    def reset(self):
        timesteps = self.sc2_env.reset()
        timestep = timesteps[0]
        obs, _, _, _ = self.convert_step(timestep)
        self.available_actions = timestep.observation['available_actions']
        return obs

    # Converts a pysc2 timestep to a Gym step.
    @staticmethod
    def convert_step(timestep: TimeStep) -> Tuple[Any, float, bool, Dict]:
        obs = timestep.observation["feature_screen"][SCREEN_FEATURES.player_relative.index]
        obs = obs.view(type=np.ndarray)  # Get a standard ndarray view instead of pysc2's subclass (NamedNumpyArray)

        # Reshape from (84, 84) to (84, 84, 1). '...' is for slicing higher-dimensional data structures and means
        # insert as many full slices (:) to extend the multi-dimensional slice to all dimensions.
        # Ref: https://stackoverflow.com/questions/118370/how-do-you-use-the-ellipsis-slicing-syntax-in-python
        obs = obs[..., np.newaxis]

        done = timestep.last()
        info = {}
        return obs, timestep.reward, done, info

    def _step(self, action) -> TimeStep:
        # Get coords by unravelling action.
        # How unravel works:
        # Ref: https://www.quora.com/What-is-a-simple-intuitive-example-for-the-unravel_index-in-Python
        coords = np.unravel_index(action, (self.screen_shape[0], self.screen_shape[1]))
        # PySC2 uses different conventions for observations (y,x) and actions (x,y)
        action = FUNCTIONS.Move_screen("now", coords[::-1])  # ::-1 reverses the tuple i.e. (1,2) becomes (2,1)

        if action.function not in self.available_actions:
            logger.warning("Attempted unavailable action {}.".format(action))
            # action = FUNCTIONS.no_op()
            action = FUNCTIONS.select_army("select")

        actions = [action]
        try:
            timestep = self.sc2_env.step(actions)[0]
        except ValueError:
            logger.error("Error occurred when attempting to execute action {}.".format(action))
            timestep = self.sc2_env.step([FUNCTIONS.no_op()])[0]

        self.available_actions = timestep.observation['available_actions']

        return timestep

    # Returns gym observation
    def step(self, action) -> Tuple[Any, float, bool, Dict]:
        timestep = self._step(action)

        if timestep.last():
            self.episode_report(timestep)

        self.elapsed_steps += 1
        return self.convert_step(timestep)

    def close(self):
        self.sc2_env.close()

    def episode_report(self, timestep: TimeStep):
        episode_score = timestep.observation["score_cumulative"][0]
        self.rolling_episode_score[self.elapsed_episodes % self.agg_n_episodes] = episode_score
        self.elapsed_episodes += 1

        if self.print_freq > 0 and self.elapsed_episodes % self.print_freq == 0:
            n_last = min(self.elapsed_episodes, self.agg_n_episodes)
            r = self.rolling_episode_score[:n_last]
            logger.info("env: %d, episode %d, score: %.1f, last %d - avg %.1f max %.1f" % (
                self.id, self.elapsed_episodes, episode_score, n_last, r.mean(), r.max()
            ))