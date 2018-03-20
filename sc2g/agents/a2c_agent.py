# ===============
# Imports
# ===============
# System / Settings / Tools
import sys
from absl import flags
from absl.flags import FLAGS
from functools import partial
# Environment
from sc2g.env.sc2gym import make_env
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
# Algorithm
from baselines.a2c.a2c import learn
from baselines.a2c.policies import CnnPolicy, LstmPolicy, LnLstmPolicy


def train():
    FLAGS(sys.argv)

    env_args = dict(
        map_name=FLAGS.map_name,
        visualize=FLAGS.visualize,
    )

    envs = SubprocVecEnv([partial(make_env, id=i, **env_args) for i in range(FLAGS.envs)])
    policy_fn = LstmPolicy

    try:
        learn(
            policy_fn,
            envs,
            seed=1,
            total_timesteps=int(1e6) * FLAGS.frames,
            lrschedule=FLAGS.lrschedule,
            nstack=1, #must be 1 for FullyConvPolicy above
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
    flags.DEFINE_string("map_name", "CollectMineralShards", "Name of the map")
    flags.DEFINE_integer("envs", 1, "Number of sc2 environments to run in parallel")
    flags.DEFINE_bool("visualize", False, "Show python visualisation")

    train()


if __name__ == "__main__":
    main()
