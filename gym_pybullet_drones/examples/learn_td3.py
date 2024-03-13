"""Script demonstrating the use of `gym_pybullet_drones`'s Gymnasium interface.

Classes HoverAviary and MultiHoverAviary are used as learning envs for the TD3 algorithm.

Example
-------
In a terminal, run as:

    $ python learn.py --multiagent false
    $ python learn.py --multiagent true

Notes
-----
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning library `stable-baselines3`.

"""
import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import TD3

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, StopTrainingOnSuccessRate, CheckpointCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = False
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin_target')
DEFAULT_ACT = ActionType('two_d_vel')
DEFAULT_AGENTS = 10
DEFAULT_NUM_OBS = 5
DEFAULT_LEVEL = 1
DEFAULT_CYLINDERS = 0
DEFAULT_MA = False

def run(multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True):

    filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    train_env = make_vec_env(MultiHoverAviary,
                                env_kwargs=dict(env_level=DEFAULT_LEVEL, num_cylinders=DEFAULT_CYLINDERS, num_drones=DEFAULT_AGENTS, num_obs = DEFAULT_NUM_OBS, obs=DEFAULT_OBS, act=DEFAULT_ACT, gui=DEFAULT_GUI),
                                n_envs=1,
                                seed=0
                                )
    # eval_env = MultiHoverAviary(env_level=DEFAULT_LEVEL, num_cylinders=DEFAULT_CYLINDERS, num_drones=DEFAULT_AGENTS, num_obs = DEFAULT_NUM_OBS, obs=DEFAULT_OBS, act=DEFAULT_ACT, gui=DEFAULT_GUI)

    #### Check the environment's spaces ########################
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    #### Train the model #######################################
    model = TD3('MlpPolicy',
                train_env,
                tensorboard_log=filename+'/tb/',
                verbose=1)

    #### Target cumulative rewards (problem-dependent) ##########
    # if DEFAULT_ACT == ActionType.ONE_D_RPM:
    #     target_reward = 474.15 if not multiagent else 100000
    # else:
    #     target_reward = 467. if not multiagent else 100000
    # callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward,
    #                                                  verbose=1)
    
    eval_callback = EvalCallback(train_env,
                                #  eval_env,
                                #  callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(10000),
                                 n_eval_episodes=10,
                                 deterministic=True,
                                 render=False)
    
    # success_callback = StopTrainingOnSuccessRate(success_rate_threshold=0.9, 
    #                                              check_freq=100, 
    #                                              verbose=1)
    
    checkpoint_path = filename+'/checkpoints/'

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=checkpoint_path,
                                             name_prefix='rl_model')
    
    # callback_list = CallbackList([eval_callback, success_callback, checkpoint_callback])
    callback_list = CallbackList([eval_callback, checkpoint_callback])
    
    model.learn(total_timesteps=int(1e7) if local else int(1e2), # shorter training in GitHub Actions pytest
                callback=callback_list,
                log_interval=100)
    
    # model.learn(total_timesteps=10000, callback=callback_list)

    #### Save the model ########################################
    model.save(filename+'/final_model.zip')
    print(filename)

    #### Print training progression ############################
    with np.load(filename+'/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

    ############################################################

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--multiagent',         default=DEFAULT_MA,            type=str2bool,      help='Whether to use example LeaderFollower instead of Hover (default: False)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
