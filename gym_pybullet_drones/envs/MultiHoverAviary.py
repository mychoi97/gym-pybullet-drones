import math
import numpy as np
import pybullet as p
import time
import random

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class MultiHoverAviary(BaseRLAviary):
    """Multi-agent RL problem: flocking."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=2,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN_TARGET,
                 act: ActionType = ActionType.TWO_D_VEL,
                #  desired_distance_flock_center=1.0,  # 3, 5, and 7 agents
                 desired_distance_flock_center=1.0,  # 10 agents
                #  desired_distance_flock_center=1.5,  # 10 agents
                #  safe_separation_distance=0.6,
                 safe_separation_distance=1.0,
                #  max_cohesion_distance=4.5,
                 max_cohesion_distance=4.0,
                 safe_to_obstacle_distance=1.0,
                 target_radius=0.5,
                 num_obs: int = 4,
                 env_level: int = 1,
                 num_cylinders: int = 0
                #  num_cylinders: int = 0
                 ):
        """Initialization of a multi-agent RL environment.

        Using the generic multi-agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.EPISODE_LEN_SEC = 100
        
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act,
                         num_obs=num_obs,
                         env_level=env_level,
                         num_cylinders=num_cylinders
                        #  num_cylinders=num_cylinders
                         )

        # self.TARGET_POS = target_pos
        # self.FREQ = freq
        self.FREQ = ctrl_freq
        # self.SPEED_THRESHOLD = speed_threshold
        self.DES_DIS_FLOCK_CENTER = desired_distance_flock_center
        self.MAX_COHESION_DISTANCE = max_cohesion_distance
        self.SAFE_SEPARATION_DISTANCE = safe_separation_distance
        self.COLLISION_DISTANCE = self.COLLISION_R * 2
        self.SAFE_TO_OBSTACLE_DISTANCE = safe_to_obstacle_distance
        self.TARGET_RADIUS = target_radius
        self.SPEED_LIMIT = 1.0

        self.SUCCESS_COUNT = 0
        self.success_list = []

    ################################ 고쳐야할 부분 ###############################################################################################################################################################

    def _computeReward(self):
        obs = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

        vel = np.zeros((1, self.NUM_DRONES, 3))
        pos = np.zeros((1, self.NUM_DRONES, 3))
        for i in range(self.NUM_DRONES):
            pos[0, i, :] = obs[i][0:3]
            vel[0, i, :] = obs[i][10:13]

        # Define the weights for the rewards

        w_nav_dis = 0.4 # Weight for navigation distance
        w_nav_ali = 0.4  # Weight for navigation alignment
        w_nav_vel = 0.0  # Weight for navigation velocity

        w_flo_coh = 0.4  # Weight for cohesion
        w_flo_sep = 0.1  # Weight for separation
        w_flo_ali = 0.1  # Weight for alignment

        w_lidar = 0.1  # Weight for lidar
            
        # w_nav_dis = 0.2 # Weight for navigation distance
        # w_nav_ali = 0.2  # Weight for navigation alignment
        # w_nav_vel = 0.0  # Weight for navigation velocity

        # w_flo_coh = 0.2  # Weight for cohesion
        # w_flo_sep = 0.2  # Weight for separation
        # w_flo_ali = 0.2  # Weight for alignment

        # w_lidar = 0.2  # Weight for lidar

        # Initialize the reward dict
        # reward = {i: 0 for i in range(self.NUM_DRONES)}
        reward = np.zeros(self.NUM_DRONES)

        for i in range(self.NUM_DRONES):

            ################################################################################
            # Compute the distance between the drone and the target
            # 점과 점 사이 거리
            distance_to_target = np.linalg.norm(pos[0, i, :] - self.TARGET_POS)

            # # x좌표 거리
            # distance_to_target =  self.TARGET_POS[0] - pos[0,i,0]

            # Compute the angle between the velocity vector of the drone and the vector to the target
            angle_to_target = np.arccos(np.dot(vel[0, i, :], (self.TARGET_POS - pos[0, i, :])) / (
                        (np.linalg.norm(self.TARGET_POS - pos[0, i, :]) * np.linalg.norm(vel[0, i, :])) + 1e-6))

            ################################################################################
            # Improved calculation for closest drones
            distances_to_drone = np.linalg.norm(pos[0, :, :] - pos[0, i, :], axis=1)
            distances_to_drone[i] = np.inf

            six_closest_indices = np.argpartition(distances_to_drone, min(self.NUM_OBS, len(distances_to_drone)-1))[:self.NUM_OBS]
            six_distances_to_drone = distances_to_drone[six_closest_indices]

            # Calculate average position of six closest drones
            all_relevant_positions = pos[0, six_closest_indices, :]
            avg_pos_six = np.mean(all_relevant_positions, axis=0)

            # Compute distance between drone i and average position of six closest drones
            distance_to_avg_pos_six = np.linalg.norm(pos[0, i, :] - avg_pos_six)

            # Calculate average velocity of six closest drones
            all_relevant_velocities = vel[0, six_closest_indices, :]
            avg_vel_six = np.mean(all_relevant_velocities, axis=0)

            # Compute angle between drone i and average velocity of six closest drones
            angle_to_avg_vel_six = np.arccos(
                np.dot(vel[0, i, :], avg_vel_six) / (np.linalg.norm(vel[0, i, :]) * np.linalg.norm(avg_vel_six) + 1e-6))

            # Compute minimum and maximum distance between drone i and the six closest drones
            min_distance_to_drone = np.min(six_distances_to_drone)

            # Load the lidar data
            lidar = self._getDroneLiDAR(i)*self.LIDAR_RANGE
            # print(f"lidar: {lidar}")

            # Compute the minimum value of the lidar data
            min_lidar = np.min(lidar)

            #################################################################################

            # Flocking 관련 reward 

            # Cohesion
            if distance_to_avg_pos_six <= self.DES_DIS_FLOCK_CENTER:
                reward[i] += w_flo_coh * 1.0
            elif distance_to_avg_pos_six > self.DES_DIS_FLOCK_CENTER and distance_to_avg_pos_six <= self.MAX_COHESION_DISTANCE:
                reward[i] += w_flo_coh * (self.DES_DIS_FLOCK_CENTER - distance_to_avg_pos_six) / (
                            self.MAX_COHESION_DISTANCE - self.DES_DIS_FLOCK_CENTER)
            else:
                reward[i] += w_flo_coh * -1.0

            # if distance_to_avg_pos_six <= self.DES_DIS_FLOCK_CENTER:
            #     reward[i] += w_flo_coh * 1.0
            # elif distance_to_avg_pos_six > self.DES_DIS_FLOCK_CENTER and distance_to_avg_pos_six <= self.MAX_COHESION_DISTANCE:
            #     reward[i] += w_flo_coh * (distance_to_avg_pos_six - self.MAX_COHESION_DISTANCE) / (
            #                 self.DES_DIS_FLOCK_CENTER - self.MAX_COHESION_DISTANCE)
            # else:
            #     reward[i] += w_flo_coh * 0.0

            # Separation
            if min_distance_to_drone <= self.COLLISION_R * 2:
                reward[i] += w_flo_sep * -1.0
            elif min_distance_to_drone > self.COLLISION_R * 2 and min_distance_to_drone <= self.SAFE_SEPARATION_DISTANCE:
                reward[i] += w_flo_sep * (min_distance_to_drone - self.SAFE_SEPARATION_DISTANCE) / (
                            self.SAFE_SEPARATION_DISTANCE - self.COLLISION_DISTANCE)
            else:
                reward[i] += w_flo_sep * 1.0
                
            # if min_distance_to_drone <= self.COLLISION_DISTANCE:
            #     reward[i] += w_flo_sep * 0.0
            # elif min_distance_to_drone > self.COLLISION_DISTANCE and min_distance_to_drone <= self.SAFE_SEPARATION_DISTANCE:
            #     reward[i] += w_flo_sep * (min_distance_to_drone - self.COLLISION_DISTANCE) / (
            #                 self.SAFE_SEPARATION_DISTANCE - self.COLLISION_DISTANCE)
            # else:
            #     reward[i] += w_flo_sep * 1.0

            # Alignment
            reward[i] += w_flo_ali * (-angle_to_avg_vel_six) / np.pi
            # reward[i] += w_flo_ali * (((-angle_to_avg_vel_six) / np.pi) + 1)

            ################################################################################
            # Obstacle 관련 reward 
            if min_lidar < self.SAFE_TO_OBSTACLE_DISTANCE:
                reward[i] += w_lidar * (min_lidar - self.SAFE_TO_OBSTACLE_DISTANCE)/(self.SAFE_TO_OBSTACLE_DISTANCE - self.COLLISION_R)
            else:
                reward[i] += w_lidar * 0.0

            # if min_lidar < self.SAFE_TO_OBSTACLE_DISTANCE:
            #     reward[i] += w_lidar * (min_lidar - self.COLLISION_R)/(self.SAFE_TO_OBSTACLE_DISTANCE - self.COLLISION_R)
            # else:
            #     reward[i] += w_lidar * 1.0

            ################################################################################

            # 타겟 도달 판별을 half plane을 사용하여 할 것임.
            # Define the half plane equation: x > target_x
            # 만약 드론의 x 좌표값이 타겟의 x 좌표값보다 크다면, 드론은 타겟의 half plane에 위치하고 있다고 판단할 수 있음.
            # if pos[0, i, 0] >= self.TARGET_POS[0]:
            if distance_to_target <= self.TARGET_RADIUS:
                reward[i] = 5.0

            # 만약 아직 드론의 x 좌표값이 타겟의 x 좌표값보다 작다면, 드론은 타겟의 half plane에 위치하지 않기 때문에 타겟 근처로 가기 위해 노력해야 함.
            else:
                # Navigation 관련 reward 

                # Navigation distance
                reward[i] += w_nav_dis * np.clip((self.prev_distance_to_target[i] - distance_to_target) / (self.SPEED_LIMIT / self.FREQ), -1, 1)
                # reward[i] += w_nav_dis * np.clip((self.prev_distance_to_target[i] - distance_to_target) / (self.SPEED_LIMIT / self.FREQ), 0, 1)

                # Navigation alignment
                reward[i] += w_nav_ali * (-angle_to_target) / np.pi
                # reward[i] += w_nav_ali * (((-angle_to_target) / np.pi) + 1)

                # Navigation velocity
                reward[i] += w_nav_vel * np.clip((np.linalg.norm(vel[0, i, :]) / self.SPEED_LIMIT), 0, 1)

            self.prev_distance_to_target[i] = distance_to_target

            ################################################################################

            # Reward Terms for terminal conditions ############################################
            # 충돌 발생
            if p.getContactPoints(bodyA=self.DRONE_IDS[i], physicsClientId=self.CLIENT):
                reward[i] = -5.0

            # 드론이 너무 멀리 떨어진 경우
            if distance_to_avg_pos_six >= self.MAX_COHESION_DISTANCE:
                reward[i] = -5.0

            # 에피소드 너무 긴 경우
            if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
                reward[i] = -5.0

        # print(f"reward: {reward}")
                
        ret = np.sum(reward)

        # return reward
        return ret

    ######################### 고쳐야 할 부분 ##########################################################################################################################################################################################################

    def _computeTerminated(self):

        bool_val = [False] * self.NUM_DRONES
        reached_target = [False] * self.NUM_DRONES
        all_distance_to_flock_center = []

        obs = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

        vel = np.zeros((1, self.NUM_DRONES, 3))
        pos = np.zeros((1, self.NUM_DRONES, 3))

        for i in range(self.NUM_DRONES):
            pos[0, i, :] = obs[i][0:3]
            vel[0, i, :] = obs[i][10:13]

        for i in range(self.NUM_DRONES):
            ################################################################################
            distance_to_target = np.linalg.norm(pos[0, i, :] - self.TARGET_POS)

            # Improved calculation for closest drones
            distances_to_drone = np.linalg.norm(pos[0, :, :] - pos[0, i, :], axis=1)
            distances_to_drone[i] = np.inf

            six_closest_indices = np.argpartition(distances_to_drone, min(self.NUM_OBS, len(distances_to_drone)-1))[:self.NUM_OBS]
            # all_indices = np.argpartition(distances_to_drone, self.NUM_DRONES-1)[:self.NUM_DRONES-1]

            # Calculate average position of six closest drones
            all_relevant_positions = pos[0, six_closest_indices, :]
            avg_pos_six = np.mean(all_relevant_positions, axis=0)

            # all_positions = pos[0, all_indices, :]
            # flock_center = np.mean(all_positions, axis=0)

            # Compute distance between drone i and average position of six closest drones
            distance_to_avg_pos_six = np.linalg.norm(pos[0, i, :] - avg_pos_six)
            # distance_to_flock_center = np.linalg.norm(pos[0, i, :] - flock_center)
            # all_distance_to_flock_center.append(distance_to_flock_center)
            all_distance_to_flock_center.append(distance_to_avg_pos_six)

            # Compute the distance between the drone and cylinders
            # distance_to_cylinders = []
            # for cylinder_pos in self.cylinder_pos:
            #     distance_to_cylinders.append(np.linalg.norm(pos - cylinder_pos))

            ############################################################################################################
            # Condition 1: Drone has crashed
            if p.getContactPoints(bodyA=self.DRONE_IDS[i], physicsClientId=self.CLIENT):
                bool_val = [True] * self.NUM_DRONES
                self.success_list.append(False)  # 실패를 리스트에 추가
                self.RESULTS['num_pass_gap'].append(False)
                self.RESULTS['average_distance_to_flock_center'].append(all_distance_to_flock_center)

                if len(self.success_list) > 10:
                    self.success_list.pop(0)
                print(f"Drone {i} crashed")
                print("Number of successes in the last 10 attempts: ", sum(self.success_list))
                print("ENV LEVEL: ", self.ENV_LEVEL)
                print("NUM CYLINDERS: ", self.NUM_CYLINDERS)
                break

            # Condition 3: Drone is too far from the average position
            if distance_to_avg_pos_six >= self.MAX_COHESION_DISTANCE:
                bool_val = [True] * self.NUM_DRONES
                self.success_list.append(False)  # 실패를 리스트에 추가
                self.RESULTS['num_pass_gap'].append(False)
                self.RESULTS['average_distance_to_flock_center'].append(all_distance_to_flock_center)
                
                if len(self.success_list) > 10:
                    self.success_list.pop(0)
                print(f"Drone {i} is too far from the average position")
                print("Number of successes in the last 10 attempts: ", sum(self.success_list))
                print("ENV LEVEL: ", self.ENV_LEVEL)
                print("NUM CYLINDERS: ", self.NUM_CYLINDERS)
                break

            # Condition 4: Drone has reached the target
            # if pos[0,i,0] >= self.WALL_POS[0]:
            # if pos[0, i, 0] >= self.TARGET_POS[0]:
            if distance_to_target <= self.TARGET_RADIUS:
                reached_target[i] = True
                print(f"Drone {i} reached target")

        # Condition 4: If all drones have reached the target
        if any(val == True for val in reached_target):
            bool_val = [True] * self.NUM_DRONES

            # Curriculum Learning을 위한 부분
            self.success_list.append(True)  # 성공을 리스트에 추가
            self.RESULTS['num_pass_gap'].append(True)
            self.RESULTS['average_distance_to_flock_center'].append(all_distance_to_flock_center)

            if len(self.success_list) > 10:  
                self.success_list.pop(0)  # 가장 오래된 결과 제거

            self.SUCCESS_COUNT = sum(self.success_list)  # 성공 횟수 계산
            print("Number of successes in the last 10 attempts: ", self.SUCCESS_COUNT)
            print("ENV LEVEL: ", self.ENV_LEVEL)
            print("NUM CYLINDERS: ", self.NUM_CYLINDERS)

            if self.SUCCESS_COUNT >= 7 and self.ENV_LEVEL < 10:  # 성공 횟수가 X 이상이면
                print("Environment goes NEXT LEVEL")
                self.ENV_LEVEL += 1
                self.init_seed()
                self.success_list = []  # 리스트 초기화

            # ENV_LEVEL이 10에 도달하면 이제 실린더 추가. 실린더 추가는 10개까지만
            if self.SUCCESS_COUNT >= 7 and self.ENV_LEVEL == 10 and self.NUM_CYLINDERS < 6:
                print("Cylinder added")
                self.NUM_CYLINDERS += 1
                self.init_seed()
                self.success_list = []  # 리스트 초기화

            
        done = {i: bool_val[i] for i in range(self.NUM_DRONES)}
        # done["__all__"] = all(bool_val)
        done = all(bool_val)

        return done

    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        # states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        # for i in range(self.NUM_DRONES):
        #     if (abs(states[i][0]) > 2.0 or abs(states[i][1]) > 2.0 or states[i][2] > 2.0 # Truncate when a drones is too far away
        #      or abs(states[i][7]) > .4 or abs(states[i][8]) > .4 # Truncate when a drone is too tilted
        #     ):
        #         return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            print("Episode is too long")
            return True
        else:
            return False

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        info = {'is_success': False}
        reached_target = [False] * self.NUM_DRONES

        obs = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

        vel = np.zeros((1, self.NUM_DRONES, 3))
        pos = np.zeros((1, self.NUM_DRONES, 3))

        for i in range(self.NUM_DRONES):
            pos[0, i, :] = obs[i][0:3]
            vel[0, i, :] = obs[i][10:13]

        for i in range(self.NUM_DRONES):
            ################################################################################
            distance_to_target = np.linalg.norm(pos[0, i, :] - self.TARGET_POS)

            # Improved calculation for closest drones
            distances_to_drone = np.linalg.norm(pos[0, :, :] - pos[0, i, :], axis=1)
            distances_to_drone[i] = np.inf

            six_closest_indices = np.argpartition(distances_to_drone, min(self.NUM_OBS, len(distances_to_drone)-1))[:self.NUM_OBS]

            # Calculate average position of six closest drones
            all_relevant_positions = pos[0, six_closest_indices, :]
            avg_pos_six = np.mean(all_relevant_positions, axis=0)

            # Compute distance between drone i and average position of six closest drones
            distance_to_avg_pos_six = np.linalg.norm(pos[0, i, :] - avg_pos_six)
           

            ############################################################################################################
            # Condition 1: Drone has crashed
            if p.getContactPoints(bodyA=self.DRONE_IDS[i], physicsClientId=self.CLIENT):
                continue

            # Condition 3: Drone is too far from the average position
            if distance_to_avg_pos_six >= self.MAX_COHESION_DISTANCE:
                continue

            # Condition 4: Drone has reached the target
            # if pos[0,i,0] >= self.WALL_POS[0]:
            # if pos[0, i, 0] >= self.TARGET_POS[0]:
            if distance_to_target <= self.TARGET_RADIUS:
                reached_target[i] = True

        if any(val == True for val in reached_target):
            info = {'is_success': True}

        return info

    ################################################################################

    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 1
        MAX_LIN_VEL_Z = 1

        MAX_PITCH_ROLL = np.pi  # Full range

        unclipped_pos_xy = state[0:2]
        unclipped_pos_z = state[2]

        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               unclipped_pos_xy,
                                               unclipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )

        unnormalized_pos_xy = unclipped_pos_xy
        unnormalized_pos_z = unclipped_pos_z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi  # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        # normalized_vel_xy = [1.0, 1.0]
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_Z
        # normalized_vel_z = 1.0
        normalized_ang_vel = state[13:16] / np.linalg.norm(state[13:16]) if np.linalg.norm(
            state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([unnormalized_pos_xy,
                                      unnormalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20, )

        return norm_and_clipped

    ################################################################################

    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.

        """
        if not (clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter,
                  "in MyFlockAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0],
                                                                                                          state[1]))
        if not (clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter,
                  "in MyFlockAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not (clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter,
                  "in MyFlockAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7],
                                                                                                         state[8]))
        if not (clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter,
                  "in MyFlockAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10],
                                                                                                          state[11]))
        if not (clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter,
                  "in MyFlockAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))