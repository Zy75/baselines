import gym
import tensorflow as tf
import cv2
from baselines.ddpg.models import Actor
from baselines.common.mpi_running_mean_std import RunningMeanStd
import numpy as np

def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std

def pi(obs):

    feed_dict = {obs0: [obs]}

    action = sess.run(actor_tf, feed_dict=feed_dict)

    action = action.flatten()
    action = np.clip(action, action_range[0], action_range[1])

    return action


with tf.device('/gpu:0'):

    epoch = 1661
    seed = 10
    n = 3000
    log_stdout = False

    observation_range=(-5., 5.)
    action_range=(-1., 1.)

    observation_shape=(15,)

    env_id = 'Reacher3d-v0'
    nb_actions = 3
    layer_norm = True

    actor = Actor(nb_actions, layer_norm=layer_norm)

    with tf.variable_scope('obs_rms'):
        obs_rms = RunningMeanStd(shape=observation_shape)

    obs0 = tf.placeholder(tf.float32, shape=(None,) + observation_shape , name='obs0')

    normalized_obs0 = tf.clip_by_value(normalize(obs0, obs_rms),
       observation_range[0], observation_range[1])

    actor_tf = actor(normalized_obs0) 

    eval_env = gym.make(env_id)
    eval_env.seed(seed)

    max_action = eval_env.action_space.high

    sess = tf.InteractiveSession()

    saver = tf.train.Saver()

    saver.restore(sess, 'model/model.ckpt-' + str(epoch))

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('model/mujoco_log' + str(epoch) + '.avi', fourcc, 50.0, (500,500))

    eval_obs = eval_env.reset()

    for _ in range(n):

        eval_action = pi(eval_obs)

        eval_new_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)

        if log_stdout:        
            print("")
            print("eval_obs=",eval_obs)
            print("eval_act=",eval_action)
            print("eval_new_obs=",eval_new_obs)
            print("eval_done=",eval_done)

        eval_obs = eval_new_obs

        out.write( cv2.cvtColor( eval_env.render(mode='rgb_array') , cv2.COLOR_BGR2RGB ) )

        if eval_done:
            eval_obs = eval_env.reset()

    out.release()

    eval_env.close()
