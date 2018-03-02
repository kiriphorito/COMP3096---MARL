# This randomly picks a pixl belonging to one of the marines.
# So this isn't selecting the marines at will but you can see that
# Randomly, it does move around quite well

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import time
import numpy
import random

# Functions
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_BUILD_SUPPLYDEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_NOOP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_MOVE_UNIT = actions.FUNCTIONS.Move_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_RALLY_UNITS_MINIMAP = actions.FUNCTIONS.Rally_Units_minimap.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

# Features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Unit IDs
_TERRAN_BARRACKS = 21
_TERRAN_COMMANDCENTER = 18
_TERRAN_SUPPLYDEPOT = 19
_TERRAN_SCV = 45
_TERRAN_MARINE = 48

# Parameters
_PLAYER_SELF = 1
_SUPPLY_USED = 3
_SUPPLY_MAX = 4
_NOT_QUEUED = [0]
_QUEUED = [1]

class SimpleAgent(base_agent.BaseAgent):
    base_top_left = None
    num_of_SCVs = 15
    num_of_processed = 0
    phase1 = False
    marine_selected = False
    marine_move = False

    def transformLocation(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]

        return [x + x_distance, y + y_distance]

    def getRandomLocation(self):
        return [numpy.random.randint(0,64), numpy.random.randint(0,64)]

    def getMarineLocation(self,unit):
        unit_type = obs.observation["screen"][_UNIT_TYPE]
        unit_y, unit_x = (unit_type == _TERRAN_MARINE).nonzero()
        position = [unit_x[unit], unit_y[unit]]
        return position

    def step(self, obs):
        super(SimpleAgent, self).step(obs)

        time.sleep(1.0)
        
        '''
        unit_type = obs.observation["screen"][_UNIT_TYPE]
        unit_y, unit_x = (unit_type == _TERRAN_MARINE).nonzero()
        marine_y, marine_x = (unit_type == _TERRAN_MARINE).nonzero()

        print(marine_y)'''




        if self.base_top_left is None:
            player_y, player_x = (obs.observation["minimap"][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            self.base_top_left = player_y.mean() <= 31

        if not self.phase1:
            if not self.marine_selected:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_MARINE).nonzero()

                randomIndex = random.randint(0, len(unit_y)-1)

                target = [unit_x[randomIndex], unit_y[randomIndex]]

                self.marine_selected = True
                self.marine_move = False

                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

            elif not self.marine_move:

                target = self.getRandomLocation()

                self.marine_selected = False
                self.marine_move = True

                self.num_of_processed = (self.num_of_processed + 1) % 2

                return actions.FunctionCall(_MOVE_UNIT, [_NOT_QUEUED, target])

        return actions.FunctionCall(_NOOP, [])
