#================================
# RESEARCH GROUP PROJECT [RGP]
#================================
# This file is part of the COMP3096 Research Group Project.

# ===============
# Imports
# ===============
# System / Settings / Tools
import sys, os
from operator import attrgetter
from absl import flags
from absl.flags import FLAGS
from functools import partial
# Environment
import sc2g
from sc2g.env.movement import MovementEnv
from sc2g.env.attack import AttackEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
# Algorithm
from baselines.a2c.a2c import learn
from baselines.a2c.policies import CnnPolicy, LstmPolicy, LnLstmPolicy
# Policies
from sc2g.policies.a2c_policy import FullyConvPolicy


def train():
    # Fetch the requested environment set in flags.
    env_class = attrgetter(FLAGS.env)(sc2g.env)

    env_args = dict(
        map_name=FLAGS.map_name,
        feature_screen_size=FLAGS.screen_size,
        feature_minimap_size=FLAGS.minimap_size,
        visualize=FLAGS.visualize,
        save_replay_episodes=FLAGS.save_replay_episodes,
        replay_dir=FLAGS.replay_dir,
    )

    envs = SubprocVecEnv([partial(env_class.make_env, id=i, **env_args) for i in range(FLAGS.envs)])

    policy_fn = CnnPolicy
    if FLAGS.policy == 'cnn':
        policy_fn = CnnPolicy
    elif FLAGS.policy == 'lstm':
        policy_fn = LstmPolicy
    elif FLAGS.policy == 'lnlstm':
        policy_fn = LnLstmPolicy
    elif FLAGS.policy == 'fullyconv':
        policy_fn = FullyConvPolicy
    else:
        print("Invalid policy function! Defaulting to {}.".format(policy_fn))

    try:
        learn(
            policy_fn,
            envs,
            seed=1,
            total_timesteps=int(1e6 * FLAGS.max_timesteps),
            lrschedule=FLAGS.lrschedule,
            ent_coef=FLAGS.entropy_weight,
            vf_coef=FLAGS.value_weight,
            max_grad_norm=1.0,
            lr=FLAGS.learning_rate
        )
    except KeyboardInterrupt:
        pass

    print("Closing environment...")
    envs.close()


def main():
    # Common
    flags.DEFINE_string("map_name", "CollectMineralShards",         "Name of the map")
    flags.DEFINE_integer("screen_size",                 32,         "Feature screen size")
    flags.DEFINE_integer("minimap_size",                32,         "Feature minimap size")
    flags.DEFINE_bool("visualize",                      False,      "Show python visualisation")
    flags.DEFINE_integer("save_replay_episodes",        500,        "How often to save replays, in episodes. 0 to disable saving replays.")
    flags.DEFINE_string("replay_dir", os.path.abspath("Replays"),   "Directory to save replays.")

    # Environment
    flags.DEFINE_string("env", "movement.MovementEnv", "Which environment to use.")

    # Algo-specific
    flags.DEFINE_integer("envs",                        2,          "Number of sc2 environments to run in parallel")
    flags.DEFINE_float("max_timesteps",                 40,         "Max timesteps, in millions")

    # Algo hyperparameters
    flags.DEFINE_string("policy",                      "fullyconv", "The policy function to use")
    flags.DEFINE_string("lrschedule",                  "constant",  "Linear or constant, learning rate schedule for baselines a2c")
    flags.DEFINE_float("learning_rate",                 3e-4,       "Learning rate")
    flags.DEFINE_float("value_weight",                  1.0,        "Value function loss weight")
    flags.DEFINE_float("entropy_weight",                1e-5,       "Entropy loss weight")

    FLAGS(sys.argv)
    print(sys.argv)

    train()


if __name__ == "__main__":
    main()
