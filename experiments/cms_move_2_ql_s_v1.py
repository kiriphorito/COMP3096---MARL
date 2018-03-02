# This randomly picks a pixl belonging to one of the marines.
# So this isn't selecting the marines at will but you can see that
# Randomly, it does move around quite well

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from sklearn.cluster import KMeans

import time
import numpy as np
import pandas as pd
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
_TERRAN_MARINE = 48

# Parameters
_PLAYER_SELF = 1
_SUPPLY_USED = 3
_SUPPLY_MAX = 4
_NOT_QUEUED = [0]
_QUEUED = [1]

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

class SimpleAgent(base_agent.BaseAgent):
    test = 0
    num_of_marines = 0
    num_of_processed = 0
    set_up_phase = False
    marine_selected = False
    marine_move = False
    marine_past_pos = []
    marine_target_pos = []
    processing_actions = False

    def __init__(self):
        super(SimpleAgent, self).__init__()

        self.qlearn = QLearningTable(actions=list(range(len(actions_set))))

        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0

        self.previous_action = None
        self.previous_state = None

    def getRandomLocation(self):
        return [np.random.randint(0,64), np.random.randint(0,64)]

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

        print("--------------- NEW STEP ---------------")

        # If all actions have been applied, make a new set of targets
        if self.num_of_processed == self.num_of_marines:
            print("Processed All Actions")
            self.num_of_processed == 0
            self.processing_actions = False

        reward = obs.reward
        #print("Reward score:" , reward)

        time.sleep(1)

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

            self.marine_target_pos = self.marine_past_pos

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
        print("Positon")
        print("Unit 0:", self.marine_past_pos[0], "  ", "Unit 1:", self.marine_past_pos[1])

        # If the actions are processed then actions need to be taken
        if not self.processing_actions:

            #print("Number of Units" , self.num_of_marines)

            for x in range(0, self.num_of_marines):
                print(x)
                target = (0,0)

                rl_action = random.randrange(0, len(actions_set) - 1)
                action_chosen = actions_set[rl_action]
                print("Unit" , x, "has chosen action:", action_chosen)

                if action_chosen == ACTION_DO_NOTHING:
                    #return actions.FunctionCall(_NOOP, [])
                    target = (self.marine_past_pos[x][0], self.marine_past_pos[x][1])
                elif action_chosen == ACTION_MOVE_UP:
                    target = (self.marine_past_pos[x][0], self.marine_past_pos[x][1] - 5)
                elif action_chosen == ACTION_MOVE_DOWN:
                    target = (self.marine_past_pos[x][0], self.marine_past_pos[x][1] + 5)
                elif action_chosen == ACTION_MOVE_LEFT:
                    target = (self.marine_past_pos[x][0] - 5, self.marine_past_pos[x][1])
                elif action_chosen == ACTION_MOVE_RIGHT:
                    target = (self.marine_past_pos[x][0] + 5, self.marine_past_pos[x][1])

                print(self.num_of_processed)

                self.marine_target_pos[x] = (target[0], target[1])

            self.test = 2096
            self.processing_actions = True
            self.num_of_processed = 0
            self.marine_selected = False
            self.marine_move = False

        print("Target")
        print(self.test)
        print("Unit 0:", self.marine_target_pos[0], "  ", "Unit 1:", self.marine_target_pos[1])

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

            self.marine_selected = False
            self.marine_move = True

            marine = self.num_of_processed
            self.num_of_processed += 1

            return actions.FunctionCall(_MOVE_UNIT, [_NOT_QUEUED, self.marine_target_pos[marine]])

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
