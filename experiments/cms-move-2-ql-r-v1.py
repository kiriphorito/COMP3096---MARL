from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import time

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
FUNCTIONS = actions.FUNCTIONS

ACTION_DO_NOTHING = 'donothing'
ACTION_MOVE_UP = 'moveup'
ACTION_MOVE_DOWN = 'movedown'
ACTION_MOVE_LEFT = 'moveleft'
ACTION_MOVE_RIGHT = 'moveright'

actions_set = [
    ACTION_DO_NOTHING,
    ACTION_MOVE_UP,
    ACTION_MOVE_DOWN,
    ACTION_MOVE_LEFT,
    ACTION_MOVE_RIGHT,
]

# From https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]

            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))

            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)

        q_predict = self.q_table.ix[s, a]
        q_target = r + self.gamma * self.q_table.ix[s_, :].max()

        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))

class CustomAgent(base_agent.BaseAgent):
  """Independent two-agent control for the CollectMineralShards minigame."""

  def __init__(self):
    super(SmartAgent, self).__init__()

    self.qlearn = QLearningTable(actions=list(range(len(actions_set))))

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

    time.sleep(1)

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
