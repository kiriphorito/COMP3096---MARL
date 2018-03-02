# This randomly picks a pixl belonging to one of the marines.
# So this isn't selecting the marines at will but you can see that
# Randomly, it does move around quite well

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from sklearn.cluster import KMeans

import time
import numpy
import random
import math

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
    num_of_marines = 0
    num_of_processed = 0
    set_up_phase = False
    marine_selected = False
    marine_move = False
    marine_past_pos = []
    marine_target_pos = [(0,0),(0,0)]

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

    def get_all_single_unit_type_pos(self, obs, unit_id, number_of_units):
        unit_type = obs.observation["screen"][_UNIT_TYPE]
        # Obtain co-ordinates of all pixels for a given unit type
        unit_y, unit_x = (unit_type == unit_id).nonzero()
        # Array of co-ordinates
        units = []
        # Fill array
        for i in range(0, len(unit_y)):
            units.append((unit_x[i], unit_y[i]))
        # Find clusters
        kmeans = KMeans(n_clusters=number_of_units)
        kmeans.fit(units)
        # Return the cluster centers
        return kmeans.cluster_centers_

    def delta_axis(self, p1, p2, axis):
        return p1[axis] - p2[axis]

    def distance(self, p1, p2):
        dx = self.delta_axis(p1, p2, 0)
        dy = self.delta_axis(p1, p2, 1)
        #print("dx:" , dx, "     dy:", dy)
        square = dx**2+dy**2
        #print("Square:" , square)
        return math.sqrt(square)

    def step(self, obs):
        super(SimpleAgent, self).step(obs)

        time.sleep(1)

        reward = obs.reward
        unit_type = obs.observation["screen"][_UNIT_TYPE]
        print(unit_type)
        print("Reward score:" , reward)

        if not self.set_up_phase:
            print("In set-up phase")
            unit_type = obs.observation["screen"][_UNIT_TYPE]
            marine_y, marine_x = (unit_type == _TERRAN_MARINE).nonzero()

            # Used to find the number of pixel allocated for each marine unit
            # There are a total of 18 pixels for 2 marines so it is 9 pixels each
            print("Marines Pixel y co-od:", marine_y)
            print("Total pixels:", len(marine_y))

            # Find the number of marines
            self.num_of_marines = int(math.ceil(len(marine_y) / 9))
            print(self.num_of_marines)

            # Get first set of co-ordinates
            position = self.get_all_single_unit_type_pos(obs, _TERRAN_MARINE, self.num_of_marines)

            for i in range(0, self.num_of_marines):
                self.marine_past_pos.append((int(position[i][0]), int(position[i][1])))
                self.marine_target_pos.append(self.marine_past_pos[i])

            print(self.marine_past_pos)

            self.set_up_phase = True

            #return actions.FunctionCall(_NOOP, [])

        # Update co-ordinates
        units_position = self.get_all_single_unit_type_pos(obs, _TERRAN_MARINE, self.num_of_marines)
        unit_position = []
        dis_to_unit = []
        for i in range(0, self.num_of_marines):
            unit_position.append((int(units_position[i][0]), int(units_position[i][1])))
        for i in range(0, self.num_of_marines):
            dis_to_unit.append(self.distance(self.marine_past_pos[i], unit_position[0]))
        if (dis_to_unit[0] < dis_to_unit[1]):
            self.marine_past_pos[0] = unit_position[0]
            self.marine_past_pos[1] = unit_position[1]
        else:
            self.marine_past_pos[0] = unit_position[1]
            self.marine_past_pos[1] = unit_position[0]

        print("Unit 0:", self.marine_past_pos[0], "  ", "Unit 1:", self.marine_past_pos[1])


        if not self.marine_selected:
            '''unit_type = obs.observation["screen"][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_MARINE).nonzero()
            randomIndex = random.randint(0, len(unit_y)-1)
            target = [unit_x[randomIndex], unit_y[randomIndex]]'''

            target = [self.marine_past_pos[self.num_of_processed][0], self.marine_past_pos[self.num_of_processed][1]]

            self.marine_selected = True
            self.marine_move = False

            return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

        elif not self.marine_move and _MOVE_UNIT in obs.observation["available_actions"]:

            target = self.getRandomLocation()
            self.marine_target_pos[self.num_of_processed] = (target[0], target[1])

            self.marine_selected = False
            self.marine_move = True

            self.num_of_processed = (self.num_of_processed + 1) % 2

            return actions.FunctionCall(_MOVE_UNIT, [_NOT_QUEUED, target])

        else:
            self.marine_move = False
            self.marine_selected = False
            self.num_of_processed = 0

        # Prediction - Under testing
        '''for i in range(0, self.num_of_marines):
            distance = distance(self.marine_past_pos[i], self.marine_target_pos[i])
            if distance < 0.5:
                continue
            delta_x = self.delta_axis(self.marine_past_pos[i], self.marine_target_pos[i], 0)
            delta_y = self.delta_axis(self.marine_past_pos[i], self.marine_target_pos[i], 1)
            normalised_x = int(delta_x/distance)
            normalised_y = int(delta_y/distance)
            predicted_x = self.marine_past_pos[i][0] + normalised_x
            predicted_y = self.marine_past_pos[i][1] + normalised_y
            predicted = (predicted_x, predicted_y)
            print(predicted)'''




        return actions.FunctionCall(_NOOP, [])
