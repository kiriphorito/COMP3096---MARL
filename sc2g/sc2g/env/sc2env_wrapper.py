#================================
# RESEARCH GROUP PROJECT [RGP]
#================================
# This file is part of the COMP3096 Research Group Project.
# This wrapper manages concurrent access to the underlying SC2 environment, allowing multiple threads to access
# the same SC2 environment in parallel.

# PySC2 Imports
from pysc2.env.sc2_marl_env import SC2MarlEnv
from pysc2.env.environment import TimeStep

# System
import threading


# Wraps a SC2MarlEnv and forwards calls to it.
class SC2EnvWrapper(SC2MarlEnv):

    def __init__(self, num_agents, **kwargs):
        super().__init__(**kwargs)

        self.num_agents = num_agents

        self.step_lock = threading.RLock()
        self.reset_lock = threading.RLock()

        self.condition_vars = [threading.Condition(), threading.Condition()]  # two Condition locks for two agents
        self.ready_vars = [False, False]
        self.current_lock = 0
        self.waiting_agents = 0  # number of agents currently waiting

        self.actions = []  # list of actions to take
        self.observations = []  # list of sc2 timesteps to return to the agents

        self.current_timesteps = None  # the global [timestep]. Use timesteps[0] to access.

        self.reset_condition = threading.Condition()
        self.reset_count = 0


    # Accepts pysc2 actions.
    def add_action(self, action):
        with self.step_lock:
            self.actions.append(action)
            if len(self.actions) >= self.num_agents:
                # OK, all agents have submitted their steps.

                # Step the SC2 environment forwards by a single step.
                self.current_timesteps = super().step(self.actions)

                # todo: perform any additional transformations to observations for individual agents
                self.observations = [self.current_timesteps for i in range(self.num_agents)]

                # Reset the action list.
                self.actions.clear()

                # Release agent 0
                with self.condition_vars[0]:
                    # print("Releasing agent 0")
                    self.ready_vars[0] = True
                    self.condition_vars[0].notify_all()

            elif self.waiting_agents == (self.num_agents - 1):
                # If we already have all other threads blocked except this one, release the next thread.
                # Release agent 1
                with self.condition_vars[1]:
                    # print("Releasing agent 1")
                    self.ready_vars[1] = True
                    self.condition_vars[1].notify_all()

    # ===============================
    # SC2 Environment Interface
    # ===============================

    def step(self, actions) -> TimeStep:
        # print("outside steplock")

        with self.step_lock:
            # print("inside steplock")
            current_lock = self.current_lock  # Remember current_lock in the currently executing scope.
            self.current_lock = 1 - self.current_lock  # Flip self.current_lock so that the next agent gets its own condition variable.

        with self.condition_vars[current_lock]:
            self.add_action(actions[0])  # NOTE: only taking into account first action from each agent
            self.waiting_agents += 1
            # print("waiting_agents +1: %d" % self.waiting_agents)
            self.condition_vars[current_lock].wait_for(lambda: self.ready_vars[current_lock])  # Suspend till this agent is notified.
            self.waiting_agents -= 1
            # print("waiting_agents -1: %d" % self.waiting_agents)
            self.ready_vars[current_lock] = False
            return self.observations[current_lock]  # Return the observation intended for this agent.

    # Called whenever a new episode starts.
    def reset(self):
        with self.reset_lock:
            if self.reset_count == 0:
                self.reset_count += 1
                self.current_timesteps = super().reset()
            elif self.reset_count == (self.num_agents - 1):
                self.reset_count = 0

            return self.current_timesteps


    def close(self):
        # todo: Are all agents done?
        print("MULTIENV REQUESTED CLOSE!")
        # super().close()
