import os
import time
from collections import deque
import pickle

from baselines.ddpg.ddpg import DDPG
from baselines.ddpg.util import mpi_mean, mpi_std, mpi_max, mpi_sum
import baselines.common.tf_util as U

from baselines import logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI

import cv2

def train(env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, param_noise, actor, critic,
    normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
    popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, memory,eval_interval,
    tau=0.01, eval_env=None, param_noise_adaption_interval=50):
    rank = MPI.COMM_WORLD.Get_rank()

    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    logger.info('scaling actions by {} before executing in env'.format(max_action))
    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale)
    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    # Set up logging stuff only for a single worker.
    if rank == 0:
        saver = tf.train.Saver()
    else:
        saver = None
    
    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)

    with U.single_threaded_session() as sess:
        # Prepare everything.
        agent.initialize(sess)

        agent.reset()
        obs = env.reset()
        if eval_env is not None:
            eval_obs = eval_env.reset()

        rollout_return_p = tf.placeholder(tf.float32, shape=(), name="roll_ret")
        rollout_Q_mean_p = tf.placeholder(tf.float32, shape=(), name="roll_Q_mean")
        eval_return_p = tf.placeholder(tf.float32, shape=(), name="eval_ret")
        eval_Q_mean_p = tf.placeholder(tf.float32, shape=(), name="eval_Q_mean")
        tr_loss_actor_p = tf.placeholder(tf.float32, shape=(), name="train_loss_actor")
        tr_loss_critic_p = tf.placeholder(tf.float32, shape=(), name="train_loss_critic")


        if rank == 0:

            with tf.name_scope('summary'):
                writer = tf.summary.FileWriter('tensorboard_log', sess.graph)
         
                tf.summary.scalar('roll_ret', rollout_return_p)
                tf.summary.scalar('roll_Q_mean', rollout_Q_mean_p)
                tf.summary.scalar('eval_ret', eval_return_p)
                tf.summary.scalar('eval_Q_mean', eval_Q_mean_p)
                tf.summary.scalar('train_loss_actor', tr_loss_actor_p)
                tf.summary.scalar('train_loss_critic', tr_loss_critic_p)
   
                merged = tf.summary.merge_all()


        done = False
        episode_reward = 0.
        episode_step = 0
        episodes = 0
        t = 0
        duration = 0

        start_time = time.time()

        for epoch in range(nb_epochs):
        
            epoch_episode_rewards = []
            epoch_episode_steps = []
            epoch_start_time = time.time()
            epoch_actions = []
            epoch_qs = []
            epoch_episodes = 0

            epoch_eval_episode_rewards = []
            epoch_eval_qs = []

            epoch_actor_losses = []
            epoch_critic_losses = []
            epoch_adaptive_distances = []

            for cycle in range(nb_epoch_cycles):

                # Perform rollouts.
                for t_rollout in range(nb_rollout_steps):
                    # Predict next action.
                    action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
                    assert action.shape == env.action_space.shape
                     
                    # Execute next action.
                    if rank == 0 and render:
                        env.render()
                    assert max_action.shape == action.shape
                    new_obs, r, done, info = env.step(max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    t += 1

                    if rank == 0 and render:
                        env.render()
                    episode_reward += r
                    episode_step += 1

                    # Book-keeping.
                    epoch_actions.append(action)
                    epoch_qs.append(q)
                    agent.store_transition(obs, action, r, new_obs, done)
                    obs = new_obs

                    if done:
                        # Episode done.
                        epoch_episode_rewards.append(episode_reward)
                        episode_rewards_history.append(episode_reward)
                        epoch_episode_steps.append(episode_step)
                        episode_reward = 0.
                        episode_step = 0
                        epoch_episodes += 1
                        episodes += 1

                        agent.reset()
                        obs = env.reset()

                # Train.
                for t_train in range(nb_train_steps):
                    # Adapt param noise, if necessary.
                    if memory.nb_entries >= batch_size and t % param_noise_adaption_interval == 0:
                        distance = agent.adapt_param_noise()
                        epoch_adaptive_distances.append(distance)

                    cl, al = agent.train()
                    epoch_critic_losses.append(cl)
                    epoch_actor_losses.append(al)
                    agent.update_target_net()

                # Evaluate.


                if eval_env is not None:
                    eval_episode_reward = 0.
                    eval_obs0 = None

                    eval_log = render_eval and epoch % eval_interval == 1 and cycle == 0

                    if eval_log:

                        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        out = cv2.VideoWriter('mujoco_log' + str(epoch) + '.avi', fourcc, 50.0, (500,500))

                    for t_rollout in range(nb_eval_steps):
                        
                        eval_action, eval_q = agent.pi(eval_obs, apply_noise=False, compute_Q=True)
                        eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                        
                        if eval_log:
                            
                            out.write( eval_env.render(mode='rgb_array') )

                            print("t=",t_rollout,"rank=",rank,"------------------------",)
                            print("eval_obs=",eval_obs0)
                            print("action=",eval_action)
                            print("eval_new_obs=",eval_obs)
                            print("reward,done=",eval_r,eval_done)
                            print(" ")
 
                            eval_obs0 = eval_obs

                        eval_episode_reward += eval_r

                        epoch_eval_qs.append(eval_q)
                        if eval_done:

                            eval_obs = eval_env.reset()
                            epoch_eval_episode_rewards.append(eval_episode_reward)
                        
                            eval_episode_rewards_history.append(eval_episode_reward)
                            eval_episode_reward = 0.
                   
                            if eval_log:
                                out.release()
 
           # Log stats.
            epoch_train_duration = time.time() - epoch_start_time
            duration = time.time() - start_time
            stats = agent.get_stats()
            combined_stats = {}
            for key in sorted(stats.keys()):
                combined_stats[key] = mpi_mean(stats[key])

            # Rollout statistics.
            combined_stats['rollout/return'] = mpi_mean(epoch_episode_rewards)
            combined_stats['rollout/return_history'] = mpi_mean(np.mean(episode_rewards_history))
            combined_stats['rollout/episode_steps'] = mpi_mean(epoch_episode_steps)
            combined_stats['rollout/episodes'] = mpi_sum(epoch_episodes)
            combined_stats['rollout/actions_mean'] = mpi_mean(epoch_actions)
            combined_stats['rollout/actions_std'] = mpi_std(epoch_actions)
            combined_stats['rollout/Q_mean'] = mpi_mean(epoch_qs)
    
            # Train statistics.
            combined_stats['train/loss_actor'] = mpi_mean(epoch_actor_losses)
            combined_stats['train/loss_critic'] = mpi_mean(epoch_critic_losses)
            combined_stats['train/param_noise_distance'] = mpi_mean(epoch_adaptive_distances)

            # Evaluation statistics.
            if eval_env is not None:

                combined_stats['eval/return'] = np.mean(epoch_eval_episode_rewards)
                combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
                combined_stats['eval/Q'] = np.mean(epoch_eval_qs)
                combined_stats['eval/episodes'] = len(epoch_eval_episode_rewards)

            # Total statistics.

            combined_stats['total/duration'] = mpi_mean(duration)
            combined_stats['total/steps_per_second'] = mpi_mean(float(t) / float(duration))
            combined_stats['total/episodes'] = mpi_mean(episodes)
            combined_stats['total/epochs'] = epoch + 1
            combined_stats['total/steps'] = t
           
            for key in sorted(combined_stats.keys()):
                logger.record_tabular(key, combined_stats[key])
            logger.dump_tabular()
            logger.info('')
            logdir = logger.get_dir()
            
            if rank == 0:
                summaryA = sess.run(merged, 
                      feed_dict={rollout_return_p: combined_stats['rollout/return'],
                      rollout_Q_mean_p: combined_stats['rollout/Q_mean'], 
                      eval_return_p: combined_stats['eval/return'],
                      eval_Q_mean_p: combined_stats['eval/Q'],
                      tr_loss_actor_p: combined_stats['train/loss_actor'],
                      tr_loss_critic_p: combined_stats['train/loss_critic'] }
                )
                writer.add_summary(summaryA,epoch)

            if rank == 0 and logdir:

                if hasattr(env, 'get_state'):
                    with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                        pickle.dump(env.get_state(), f)
                if eval_env and hasattr(eval_env, 'get_state'):
                    with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                        pickle.dump(eval_env.get_state(), f)
