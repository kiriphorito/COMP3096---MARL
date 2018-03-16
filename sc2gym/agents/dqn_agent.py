#================================
# RESEARCH GROUP PROJECT [RGP]
#================================
# This file is part of the COMP3096 Research Group Project.

# System imports
import sys
from absl import flags

# Environment imports
import gym
import sc2gym.envs # Import SC2CollectMineralShards-v0

# Algorithm imports
from baselines import deepq

# Alias setup
FLAGS = flags.FLAGS

def main():
    FLAGS(sys.argv)

    map_name = 'CollectMineralShards'
    env = gym.make("SC2CollectMineralShards-v0")

    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True
    )

    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-5,
        max_timesteps=100000000,
        buffer_size=100000,
        exploration_fraction=0.5,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=100000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=True
    )

    try:
        print("Saving model to {}_model.pkl".format(map_name))
        act.save("{}_model.pkl".format(map_name))
    except:
        print("Error saving model.")

    print("Saving replay...")
    env.save_replay(map_name)

    print("Closing environment...")
    env.close()


if __name__ == "__main__":
    main()