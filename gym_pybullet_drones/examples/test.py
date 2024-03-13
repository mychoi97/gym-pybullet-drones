import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, StopTrainingOnSuccessRate, CheckpointCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

output_folder = "/home/mychoi/Research/Quadrotor/gym-pybullet-drones/gym_pybullet_drones/examples/results"
filename = os.path.join(output_folder, "save-02.07.2024_21.53.07")
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin_target')
DEFAULT_ACT = ActionType('two_d_vel')
DEFAULT_AGENTS = 6
DEFAULT_NUM_OBS = 5
# DEFAULT_NUM_CYLINDERS = 0
DEFAULT_PLOT = True
DEFAULT_LEVEL = 10
DEFAULT_CYLINDERS = 0

# if os.path.isfile(filename+'/best_model.zip'):
#     path = filename+'/best_model.zip'
if os.path.isfile(filename+'/checkpoints/rl_model_5080000_steps.zip'):
    path = filename+'/checkpoints/rl_model_5080000_steps.zip'
else:
    print("[ERROR]: no model under the specified path", filename)
# model = PPO.load(path)
model = TD3.load(path)

#### Show (and record a video of) the model's performance ##
test_env = MultiHoverAviary(gui=True,
                                num_drones=DEFAULT_AGENTS,
                                num_obs = DEFAULT_NUM_OBS, 
                                # num_cylinders = DEFAULT_NUM_CYLINDERS,
                                obs=DEFAULT_OBS,
                                act=DEFAULT_ACT,
                                env_level=DEFAULT_LEVEL,
                                num_cylinders=DEFAULT_CYLINDERS,
                                record=DEFAULT_RECORD_VIDEO)
test_env_nogui = MultiHoverAviary(num_drones=DEFAULT_AGENTS, num_obs = DEFAULT_NUM_OBS, obs=DEFAULT_OBS, act=DEFAULT_ACT)
# logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
#             num_drones=DEFAULT_AGENTS,
#             output_folder=output_folder,
#             colab=DEFAULT_COLAB
#             )

mean_reward, std_reward = evaluate_policy(model,
                                            test_env,
                                            n_eval_episodes=10
                                            )
print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

obs, info = test_env.reset(seed=42, options={})
obs = obs[0] if isinstance(obs, tuple) else obs

start = time.time()
for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
    action, _states = model.predict(obs,
                                    deterministic=True
                                    )
    obs, reward, terminated, truncated, info = test_env.step(action)
    obs = obs[0] if isinstance(obs, tuple) else obs

    obs2 = obs.squeeze()
    act2 = action.squeeze()
    print("Obs:", obs, "\tAction:", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
    # if DEFAULT_OBS == ObservationType.KIN_TARGET:
    #     for d in range(DEFAULT_AGENTS):
    #         logger.log(drone=d,
    #             timestamp=i/test_env.CTRL_FREQ,
    #             state=np.hstack([obs2[d][0:3],
    #                                 np.zeros(4),
    #                                 obs2[d][3:15],
    #                                 act2[d]
    #                                 ]),
    #             control=np.zeros(12)
    #             )
    test_env.render()
    print(terminated)
    sync(i, start, test_env.CTRL_TIMESTEP)

    if terminated:
        obs = test_env.reset(seed=42, options={})
        obs = obs[0] if isinstance(obs, tuple) else obs

test_env.close()

# if DEFAULT_PLOT and DEFAULT_OBS == ObservationType.KIN_TARGET:
#     logger.plot()
