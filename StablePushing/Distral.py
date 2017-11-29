from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
from baselines.common import Dataset, explained_variance, fmt_row, zipsame
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
import os.path as osp
import gym, logging
from baselines import logger
import sys
from StablePushingEnv import *
from TestEnv import *
from pposimple import *
from TestEnv_2 import *
from TestEnv_1 import *

def build_mlp(
        input_placeholder, 
        output_size,
        scope, 
        n_layers=2, 
        size=64, 
        activation=tf.tanh,
        output_activation=None
        ):

    with tf.variable_scope(scope):
        # YOUR_CODE_HERE
        h = tf.layers.dense(input_placeholder, size, activation=activation)
        for i in range(n_layers):
            h = tf.layers.dense(h, size, activation=activation)
        y = tf.layers.dense(h, output_size, activation=output_activation)
        return y

def train(num_timesteps, iters):
    from baselines.ppo1 import mlp_policy
    U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env0 = TestEnv()
    model_0 = learn(env0, policy_fn, "pi0", 
            max_timesteps=num_timesteps,
            timesteps_per_batch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',
        )
    # env0.close()

    env1 = TestEnv1()
    model_1 = learn(env1, policy_fn, "pi1", 
            max_timesteps=num_timesteps,
            timesteps_per_batch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',
        )
    # env1.close()

    env2 = TestEnv2()
    model_2 = learn(env2, policy_fn, "pi2", 
            max_timesteps=num_timesteps,
            timesteps_per_batch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',
        )

    ob_space = env0.observation_space
    ac_space = env0.action_space
    pi = policy_fn("model_d", ob_space, ac_space) # Construct network for new policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None])
    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[])

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])
    kl = pi.pd.kl(model_0.pd) + pi.pd.kl(model_1.pd) + pi.pd.kl(model_2.pd)
    ent = model_0.pd.entropy() + model_1.pd.entropy() + model_2.pd.entropy()
    meankl = U.mean(kl)
    meanent = U.mean(ent)
    loss = - meankl # - U.mean(tf.exp(model_0.pd.logp(ac)) * atarg) - U.mean(tf.exp(model_1.pd.logp(ac)) * atarg) - U.mean(tf.exp(model_2.pd.logp(ac)) * atarg)
    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], loss + [U.flatgrad(loss, var_list)])
    adam = MpiAdam(var_list, epsilon=1e-5)
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], loss)

    U.initialize()
    adam.sync()

    seg_gen0 = traj_segment_generator(model_0, env0, 1000, stochastic=True)
    seg_gen1 = traj_segment_generator(model_1, env1, 1000, stochastic=True)
    seg_gen2 = traj_segment_generator(model_2, env2, 1000, stochastic=True)

    seg_gend0 = traj_segment_generator(pi, env0, 1000, stochastic=True)
    seg_gend1 = traj_segment_generator(pi, env1, 1000, stochastic=True)
    seg_gend2 = traj_segment_generator(pi, env2, 1000, stochastic=True)

    lenbuffer0 = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer0 = deque(maxlen=100)
    lenbuffer1 = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer1 = deque(maxlen=100)
    lenbuffer2 = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer2 = deque(maxlen=100)

    rew0 = []
    rew1 = []
    rew2 = []

    # env2.close()
    # return model_0, model_1, model_2
    for i in range(iters):

        logger.log("********** Iteration %i ************"%i)
        cur_lrmult = 1.0

        seg0 = seg_gen0.__next__()
        add_vtarg_and_adv(seg0, 0.99, 0.95)

        ob, ac, atarg, tdlamret = seg0["ob"], seg0["ac"], seg0["adv"], seg0["tdlamret"]
        vpredbefore = seg0["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret))
        optim_batchsize = ob.shape[0]

        for _ in range(10):
            # losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                adam.update(g, 3e-4 * cur_lrmult)

        seg1 = seg_gen1.__next__()
        add_vtarg_and_adv(seg1, 0.99, 0.95)

        ob, ac, atarg, tdlamret = seg1["ob"], seg1["ac"], seg1["adv"], seg1["tdlamret"]
        vpredbefore = seg1["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret))
        optim_batchsize = ob.shape[0]

        for _ in range(10):
            # losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                adam.update(g, 3e-4 * cur_lrmult)

        seg2 = seg_gen2.__next__()
        add_vtarg_and_adv(seg2, 0.99, 0.95)

        ob, ac, atarg, tdlamret = seg2["ob"], seg2["ac"], seg2["adv"], seg2["tdlamret"]
        vpredbefore = seg2["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret))
        optim_batchsize = ob.shape[0]

        for _ in range(10):
            # losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                adam.update(g, 3e-4 * cur_lrmult)

        segd0 = seg_gend0.__next__()
        segd1 = seg_gend1.__next__()
        segd2 = seg_gend2.__next__()

        lrlocal0 = (segd0["ep_lens"], segd0["ep_rets"]) # local values
        listoflrpairs0 = MPI.COMM_WORLD.allgather(lrlocal0) # list of tuples
        lens0, rews0 = map(flatten_lists, zip(*listoflrpairs0))
        lenbuffer0.extend(lens0)
        rewbuffer0.extend(rews0)
        mean_rew0 = np.mean(rewbuffer0)
        logger.record_tabular("Env0EpLenMean", np.mean(lenbuffer0))
        logger.record_tabular("Env0EpRewMean", mean_rew0)
        rew0.append(mean_rew0)

        lrlocal1 = (segd1["ep_lens"], segd1["ep_rets"]) # local values
        listoflrpairs1 = MPI.COMM_WORLD.allgather(lrlocal1) # list of tuples
        lens1, rews1 = map(flatten_lists, zip(*listoflrpairs1))
        lenbuffer1.extend(lens1)
        rewbuffer1.extend(rews1)
        mean_rew1 = np.mean(rewbuffer1)
        logger.record_tabular("Env1EpLenMean", np.mean(lenbuffer1))
        logger.record_tabular("Env1EpRewMean", mean_rew1)
        rew1.append(mean_rew1)

        lrlocal2 = (segd2["ep_lens"], segd2["ep_rets"]) # local values
        listoflrpairs2 = MPI.COMM_WORLD.allgather(lrlocal2) # list of tuples
        lens2, rews2 = map(flatten_lists, zip(*listoflrpairs2))
        lenbuffer2.extend(lens2)
        rewbuffer2.extend(rews2)
        mean_rew2 = np.mean(rewbuffer2)
        logger.record_tabular("Env2EpLenMean", np.mean(lenbuffer2))
        logger.record_tabular("Env2EpRewMean", mean_rew2)
        rew2.append(mean_rew2)


        if MPI.COMM_WORLD.Get_rank()==0:
            logger.dump_tabular()

    return model_0, model_1, model_2, pi, np.array(rew0), np.array(rew1), np.array(rew2)

    	

