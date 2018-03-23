#================================
# RESEARCH GROUP PROJECT [RGP]
#================================
# This file is part of the COMP3096 Research Group Project.

# ===============
# Imports
# ===============
# System / Settings
import sys, os, datetime
from operator import attrgetter
from absl import flags
from absl.flags import FLAGS
# Environment
import sc2g
from sc2g.env.movement import MovementEnv, DirectedMovementEnv, MultiMovementEnv
# Algorithm
from baselines import deepq
from baselines.common.atari_wrappers import FrameStack

# Globals
save_model_freq = 0


def train():
    # Fetch the requested environment set in flags.
    env_class = attrgetter(FLAGS.env)(sc2g.env)

    env = env_class.make_env(
        map_name=FLAGS.map_name,
        feature_screen_size=FLAGS.screen_size,
        feature_minimap_size=FLAGS.minimap_size,
        visualize=FLAGS.visualize,
        save_replay_episodes=FLAGS.save_replay_episodes,
        replay_dir=FLAGS.replay_dir,
    )

    # Stack frames (memory optimisation)
    if FLAGS.num_stack_frames > 0:
        env = FrameStack(env, FLAGS.num_stack_frames)

    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True
    )

    act = deepq.learn(
        env,
        q_func=model,
        lr=FLAGS.learning_rate,                 # Learning rate for adam optimizer
        max_timesteps=FLAGS.max_timesteps,      # Max timesteps
        buffer_size=FLAGS.buffer_size,          # Size of replay buffer
        exploration_fraction=FLAGS.exploration_fraction,    # Fraction of max_timesteps over which exploration rate is annealed
        exploration_final_eps=FLAGS.exploration_final_eps,  # Final value of random action probability
        train_freq=FLAGS.train_freq,            # How often the model is updated, in steps
        print_freq=FLAGS.print_freq,            # How often training progress is printed, in episodes
        checkpoint_freq=FLAGS.checkpoint_freq,  # How often to save the model, in steps
        learning_starts=FLAGS.learning_starts,  # How many steps before learning starts
        gamma=FLAGS.gamma,                      # Discount factor
        target_network_update_freq=FLAGS.target_network_update_freq,  # How often the target network is updated
        prioritized_replay=FLAGS.prioritized_replay,
        callback=deepq_callback,
    )

    print("Saving model...")
    save_model(act)

    print("Saving replay...")
    env.unwrapped.sc2_env.save_replay(FLAGS.map_name)

    print("Closing environment...")
    env.close()


# This callback is called every timestep of training.
def deepq_callback(locals, globals):
    if save_model_freq > 0:
        t = locals["t"]
        if (t % save_model_freq == 0) and (t > 0):
            print("Saving model at timestep %d..." % t)
            save_model(locals["act"])


def save_model(act):
    try:
        now = datetime.datetime.now()
        filename = "{}_model_{}.pkl".format(FLAGS.map_name, now.strftime("%Y-%m-%d_%H-%M"))
        act.save(filename)
        print("Saved model to {}".format(filename))
    except:
        print("Error saving model.")


def main():
    # Common
    flags.DEFINE_string("map_name", "CollectMineralShards",         "Name of the map")
    flags.DEFINE_integer("screen_size",                 84,         "Feature screen size")
    flags.DEFINE_integer("minimap_size",                64,         "Feature minimap size")
    flags.DEFINE_bool("visualize",                      False,      "Show python visualisation")
    flags.DEFINE_integer("save_replay_episodes",        500,        "How often to save replays, in episodes. 0 to disable saving replays.")
    flags.DEFINE_string("replay_dir", os.path.abspath("Replays"),   "Directory to save replays, relative to the current working directory.")

    # Environment
    flags.DEFINE_string("env", "movement.DirectedMovementEnv", "Which environment to use.")

    # Algo-specific settings
    flags.DEFINE_integer("print_freq",                  10,         "How often training progress is printed, in episodes")  # 100
    flags.DEFINE_integer("checkpoint_freq",             10000,      "How often to checkpoint the model (in temporary directory), in steps")  # 10000
    flags.DEFINE_integer("save_model_freq",             250000,     "How often to save the model, in steps")
    flags.DEFINE_integer("num_stack_frames",            4,          "Number of frames to stack together (memory optimisation). Set 0 to disable stacking.")

    # Algo hyperparameters
    flags.DEFINE_float("learning_rate",                 1e-5,       "Learning rate for adam optimizer") # 5e-4
    flags.DEFINE_integer("max_timesteps",               2000000,    "Max timesteps")  # 100000
    flags.DEFINE_integer("buffer_size",                 100000,     "Size of replay buffer")  # 50000
    flags.DEFINE_float("exploration_fraction",          0.5,        "Fraction of max_timesteps over which exploration rate is annealed")  # 0.1
    flags.DEFINE_float("exploration_final_eps",         0.01,       "Final value of random action probability")  # 0.02
    flags.DEFINE_integer("train_freq",                  4,          "How often the model is updated, in steps")  # 1
    flags.DEFINE_integer("learning_starts",             100000,     "How many steps before learning starts")  # 1000
    flags.DEFINE_float("gamma",                         0.99,       "Discount factor")  # 1.0
    flags.DEFINE_integer("target_network_update_freq",  500,        "How often the target network is updated")  # 500
    flags.DEFINE_bool("prioritized_replay",             True,       "Whether prioritized replay is used")  # True

    FLAGS(sys.argv)
    global save_model_freq  # Make this global since it's checked every timestep
    save_model_freq = FLAGS.save_model_freq

    train()


if __name__ == "__main__":
    main()
