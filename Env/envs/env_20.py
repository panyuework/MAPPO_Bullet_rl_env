# Copyright 2023 Shathushan Sivashangaran

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Authors: Shathushan Sivashangaran, Apoorva Khairnar

import gym
import numpy as np
import math
import pybullet as p
import matplotlib.pyplot as plt
import pandas as pd
import random
from Env.resources.x10car_v1 import x10car_v1
from Env.resources.x10car_v0 import x10car_v0
from Env.resources.x_mycar import x_mycar
from Env.resources.x_mycar2 import x_mycar2

from Env.resources.Walls_20 import walls_20
from Env.resources.Walls_50 import walls_50

from Env.resources.Ground_20 import ground_20
from Env.resources.Ground_50 import ground_50

from Env.resources.Rock_1 import rock_1
from Env.resources.Rock_2 import rock_2
from Env.resources.Rock_3 import rock_3
from Env.resources.Rock_4 import rock_4
from Env.resources.Rock_5 import rock_5
from Env.resources.Rock_6 import rock_6
from Env.resources.Rock_7 import rock_7
from Env.resources.ball1 import ball1
from Env.resources.ball2 import ball2
from Env.resources.ball3 import ball3
from Env.resources.ball4 import ball4
from Env.resources.apple import apple
from Env.envs.envs.agents import RaceCar
from Env.envs.envs.agents import Drone
from Env.envs.envs.agents import MJCFAgent
from Env.resources.Tree_1 import tree_1
from Env.resources.Tree_2 import tree_2
from Env.resources.Tree_3 import tree_3
from Env.resources.Tree_4 import tree_4
DISCRETE_ACTIONS = {
    # # coast
    # 0: [0.0, 0.0],
    # # turn left
    # 1: [0.0, -0.5],
    # # turn right
    # 2: [0.0, 0.5],
    # # forward
    # 3: [1.0, 0.0],
    # # brake
    # 4: [-0.5, 0.0],
    # # forward left
    # 5: [0.5, -0.05],
    # # forward right
    # 6: [0.5, 0.05],
    # # brake left
    # 7: [-0.5, -0.5],
    # # brake right
    # 8: [-0.5, 0.5],
    # coast
    0: [0.5, 0.5],
    # turn left
    1: [0.5, -0.5],
    # turn right
    2: [1.0, 0.2],
    # forward
    3: [1.0, -0.2],
    # brake
    4: [1.0, 0.0],
    # forward left
    5: [1.0, -0.05],
    # forward right
    6: [1.0, 0.05],
    # brake left
    7: [0.5, -0.1],
    # brake right
    8: [0.5, 0.1],
}
class X10Car_Env_out20(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32))

        distlow = np.zeros(100)
        disthigh = [1000]*100
        self.agent_num = 3
        # self.apple_true = [1] * 16
        # self.bomb_true = [1] * 8

        # Observation space for 90 deg FoV lidar with 100 points. For alternate FoV and number of points modify get_distance() in the x10car file

        self.observation_space = gym.spaces.Box(
            low=np.array(distlow, dtype=np.float32),
            high=np.array(disthigh, dtype=np.float32))
        self.np_random, _ = gym.utils.seeding.np_random()

        self.client = p.connect(p.DIRECT, options="--gpu")
        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1/50, self.client)
        # 设置相机位置
        camera_distance = 15  # 相机距离目标点的距离
        camera_yaw = 150  # 相机水平旋转角度
        camera_pitch = -45  # 相机垂直旋转角度
        camera_target_position = [0, 0, 0]  # 相机目标点位置，设为场景中心点或感兴趣区域的坐标
        self.num_step = 200
        self.steps = 0
        # 设置相机视角
        p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_position)
        self.car = None
        self.wall = None
        self.plane = None
        self.rock_1 = None
        self.rock_2 = None
        self.rock_3 = None
        self.rock_4 = None
        self.rock_5 = None
        self.rock_6 = None
        self.rock_7 = None
        self.tree_1 = None
        self.tree_2 = None
        self.tree_3 = None
        self.tree_4 = None

        self.done = False
        self.prev_dist_to_wall = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.trajectory = pd.DataFrame({'Agent Steps':[], 'Car Observation X':[], 'Car Observation Y':[], 'Minimum Distance':[], 'Reward':[]})
        self.writeReward = pd.DataFrame({'Reward':[]})
        self.store_agent_steps = 0
        self.store_reward = 0
        self.episode_reward = 0
        self.writeEpisodeReward = pd.DataFrame({'Episode Reward':[]})
        self.reset()

    def step(self, action):
        action0 = np.argmax(action[0])
        action0 = DISCRETE_ACTIONS[int(action0)]
        # action0 = DISCRETE_ACTIONS[action[0]]
        action1 = np.argmax(action[1])
        action1 = DISCRETE_ACTIONS[int(action1)]
        action2 = np.argmax(action[2])
        action2 = DISCRETE_ACTIONS[int(action2)]
        # action1 = DISCRETE_ACTIONS[action[1]]
        # action2 = DISCRETE_ACTIONS[action[2]]

        self.car0.apply_action(action0)
        self.car1.apply_action(action1)# perform action
        self.car2.apply_action(action2)

        p.stepSimulation()


#################################obs#############################################3
        car_ob = []
        # task_ob = []
        # dist_to_wall = []
        car0_rgb = self.car0.get_camImg(self.car0.x10car_v0, self.client)  # [100,100,4]
        car0_ob = self.car0.get_observation()  # 获取车辆的状态list(7)
        car0_dist_to_wall = self.car0.get_distance()  # 获取距离墙壁的距离list(93)
        combined_list = car0_ob + car0_dist_to_wall
        # 将列表转换为NumPy数组
        arr = np.array(combined_list)
        # 使用reshape函数改变数组的形状
        reshaped_arr = np.repeat(arr[:, np.newaxis], 100, axis=1)[:, :, np.newaxis]
        ob_0 = np.concatenate((car0_rgb, reshaped_arr), axis=2)
        car_ob.append(ob_0)

        car1_rgb = self.car1.get_camImg(self.car1.x10car_v0, self.client)  # [100,100,4]
        car1_ob = self.car1.get_observation()  # 获取车辆的状态list(7)
        car1_dist_to_wall = self.car1.get_distance()  # 获取距离墙壁的距离list(93)
        combined_list = car1_ob + car1_dist_to_wall
        # 将列表转换为NumPy数组
        arr = np.array(combined_list)
        # 使用reshape函数改变数组的形状
        reshaped_arr = np.repeat(arr[:, np.newaxis], 100, axis=1)[:, :, np.newaxis]
        ob_1 = np.concatenate((car1_rgb, reshaped_arr), axis=2)
        car_ob.append(ob_1)

        car2_rgb = self.car2.get_camImg(self.car2.x10car_v0, self.client)  # [100,100,4]
        car2_ob = self.car2.get_observation()  # 获取车辆的状态list(7)
        car2_dist_to_wall = self.car2.get_distance()  # 获取距离墙壁的距离list(93)
        combined_list = car2_ob + car2_dist_to_wall
        # 将列表转换为NumPy数组
        arr = np.array(combined_list)
        # 使用reshape函数改变数组的形状
        reshaped_arr = np.repeat(arr[:, np.newaxis], 100, axis=1)[:, :, np.newaxis]
        ob_2 = np.concatenate((car2_rgb, reshaped_arr), axis=2)
        car_ob.append(ob_2)

######################################reward#######################################
        reward = []
        self.car0_reward = 0
        self.car1_reward = 0
        self.car2_reward = 0

        min_dist_to_wall0 = min(car0_dist_to_wall)
        min_dist_to_wall1 = min(car1_dist_to_wall)
        min_dist_to_wall2 = min(car2_dist_to_wall)

                        ###############reward_car0#############
        # car0_reward, dis0_ball1 = self.car0.cul_reward(action[0], min_dist_to_wall0, car0_ob, )
        # car0_eat_apple = self.find_coordinates_within_distance(self.list_car[0], self.list_apple)
        # car0_eat_bomb = self.find_coordinates_within_distance(self.list_car[0], self.list_bomb)
        car0_reward = self.compute_car0_reward()
                          ###############reward_car1#############
        # car1_reward, dis1_ball1 = self.car1.cul_reward(action[1], min_dist_to_wall1, car1_ob, )
        car1_reward = self.compute_car1_reward()
                            ###############reward_car2#############
        # car2_reward, dis2_ball1 = self.car2.cul_reward(action[2], min_dist_to_wall2, car2_ob, )
        car2_reward = self.compute_car2_reward()
        private = 0.658
        unprivate = 1 - private
        reward0 = private * car0_reward + unprivate * (car1_reward + car2_reward)
        # reward.append(reward0)
        reward.append(reward0)

        reward1 = private * car1_reward + unprivate * (car0_reward + car2_reward)
        # reward.append(reward1)
        reward.append(reward1)

        reward2 = private * car2_reward + unprivate * (car0_reward + car1_reward)
        # reward.append(reward2)
        reward.append(reward2)

#####################################done############################
        if (self.num_step == self.steps+1):
            self.done = True
            self.steps = 0
        self.steps = self.steps + 1

        return car_ob, reward, self.done, dict()
        # return ob, reward,False, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.81)
        # Reload environment assets
        self.plane = ground_20(self.client)
        self.wall = walls_20(self.client)

        # 生成23个坐标点
        points = np.random.uniform(low=[-9, -9, 0], high=[9, 9, 0], size=(27, 3))
        self.apple_true = [1] * 16
        self.bomb_true = [1] * 8
        # 第一个列表随机取3个点
        indices1 = np.random.choice(27, size=3, replace=False)
        list1 = list(points[indices1])
        self.list_car = list1

        # 剩下的点
        mask = np.ones(27, dtype=bool)
        mask[indices1] = False
        remaining_points = points[mask]

        # 第二个列表随机取8个点
        indices2 = np.random.choice(len(remaining_points), size=8, replace=False)
        list2 = list(remaining_points[indices2])
        self.list_bomb = list2

        # 剩下的点
        mask = np.ones(len(remaining_points), dtype=bool)
        mask[indices2] = False
        remaining_points = remaining_points[mask]

        # 第三个列表包含剩下的12个点
        list3 = list(remaining_points)
        self.list_apple = list3

        self.car0 = x_mycar2(self.client, list1[0])
        self.car1 = x_mycar2(self.client, list1[1])
        self.car2 = x_mycar2(self.client, list1[2])
        # self.car4 = RaceCar(self.client)
        # self.air = Drone(self.client)
        # self.M = MJCFAgent(self.client)


###################apple#######################

        self.rock_0 = rock_1(self.client, list3[0])
        self.rock_1 = rock_1(self.client, list3[1])
        self.rock_2 = rock_1(self.client, list3[2])
        self.rock_3 = rock_1(self.client, list3[3])
        self.rock_4 = rock_1(self.client, list3[4])
        self.rock_5 = rock_1(self.client, list3[5])
        self.rock_6 = rock_1(self.client, list3[6])
        self.rock_7 = rock_1(self.client, list3[7])
        self.rock_8 = rock_1(self.client, list3[8])
        self.rock_9 = rock_1(self.client, list3[9])
        self.rock_10= rock_1(self.client, list3[10])
        self.rock_11= rock_1(self.client, list3[11])
        self.rock_12= rock_1(self.client, list3[12])
        self.rock_13= rock_1(self.client, list3[13])
        self.rock_14 = rock_1(self.client, list3[14])
        self.rock_15 = rock_1(self.client, list3[15])

        # self.rock_16 = rock_1(self.client, list3[16])
        # self.rock_17 = rock_1(self.client, list3[17])
        # self.rock_18 = rock_1(self.client, list3[18])
        # self.rock_19 = rock_1(self.client, list3[19])
        # self.rock_20 = rock_1(self.client, list3[20])
        # self.rock_21 = rock_1(self.client, list3[21])
        # self.rock_22 = rock_1(self.client, list3[22])
        # self.rock_23 = rock_1(self.client, list3[23])
        # self.rock_24 = rock_1(self.client, list3[24])
        # self.rock_25 = rock_1(self.client, list3[25])
        # self.rock_26 = rock_1(self.client, list3[26])
        # self.rock_27 = rock_1(self.client, list3[27])
        # self.rock_28 = rock_1(self.client, list3[28])
        # self.rock_29 = rock_1(self.client, list3[29])
        # self.rock_30 = rock_1(self.client, list3[30])
        # self.rock_31 = rock_1(self.client, list3[31])
        # self.rock_8 = rock_1(self.client)
#########################bomb#################
        self.ball_0 = ball1(self.client, list2[0])
        self.ball_1 = ball1(self.client, list2[1])
        self.ball_2 = ball1(self.client, list2[2])
        self.ball_3 = ball1(self.client, list2[3])
        self.ball_4 = ball1(self.client, list2[4])
        self.ball_5 = ball1(self.client, list2[5])
        self.ball_6 = ball1(self.client, list2[6])
        self.ball_7 = ball1(self.client, list2[7])

        # self.ball_8 = ball1(self.client, list2[8])
        # self.ball_9 = ball1(self.client, list2[9])
        # self.ball_10 = ball1(self.client, list2[10])
        # self.ball_11= ball1(self.client, list2[11])
        # self.ball_12= ball1(self.client, list2[12])
        # self.ball_13= ball1(self.client, list2[13])
        # self.ball_14= ball1(self.client, list2[14])
        # self.ball_15= ball1(self.client, list2[15])
        # self.apple = apple(self.client)
        # self.tree_1 = tree_1(self.client)
        # self.tree_2 = tree_2(self.client)
        # self.tree_3 = tree_3(self.client)
        # self.tree_4 = tree_4(self.client)

        self.done = False

        # self.prev_dist_to_wall = self.car0.get_distance()
        # a, b = self.car.get_ids(self)
#################################obs#############################################3
        car_ob = []
        # task_ob = []
        # dist_to_wall = []
        car0_rgb = self.car0.get_camImg(self.car0.x10car_v0, self.client) # [100,100,4]
        car0_ob = self.car0.get_observation()  # 获取车辆的状态list(7)
        car0_dist_to_wall = self.car0.get_distance()  # 获取距离墙壁的距离list(93)
        combined_list = car0_ob + car0_dist_to_wall
        # 将列表转换为NumPy数组
        arr = np.array(combined_list)
        # 使用reshape函数改变数组的形状
        reshaped_arr = np.repeat(arr[:, np.newaxis], 100, axis=1)[:, :, np.newaxis]
        ob_0 = np.concatenate((car0_rgb, reshaped_arr), axis=2)
        car_ob.append(ob_0)

        car1_rgb = self.car1.get_camImg(self.car1.x10car_v0, self.client)  # [100,100,4]
        car1_ob = self.car1.get_observation()  # 获取车辆的状态list(7)
        car1_dist_to_wall = self.car1.get_distance()  # 获取距离墙壁的距离list(93)
        combined_list = car1_ob + car1_dist_to_wall
        # 将列表转换为NumPy数组
        arr = np.array(combined_list)
        # 使用reshape函数改变数组的形状
        reshaped_arr = np.repeat(arr[:, np.newaxis], 100, axis=1)[:, :, np.newaxis]
        ob_1 = np.concatenate((car1_rgb, reshaped_arr), axis=2)
        car_ob.append(ob_1)

        car2_rgb = self.car2.get_camImg(self.car2.x10car_v0, self.client)  # [100,100,4]
        car2_ob = self.car2.get_observation()  # 获取车辆的状态list(7)
        car2_dist_to_wall = self.car2.get_distance()  # 获取距离墙壁的距离list(93)
        combined_list = car2_ob + car2_dist_to_wall
        # 将列表转换为NumPy数组
        arr = np.array(combined_list)
        # 使用reshape函数改变数组的形状
        reshaped_arr = np.repeat(arr[:, np.newaxis], 100, axis=1)[:, :, np.newaxis]
        ob_2 = np.concatenate((car2_rgb, reshaped_arr), axis=2)
        car_ob.append(ob_2)


        # self.car0.get_camImg(self.car0.x10car_v0, self.client)
        return car_ob


    def render(self, mode='human'):
        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((100, 100, 4)))

        # Base information
        car_id, client_id = self.car.get_ids()
        wall_id = self.wall.get_ids()

        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                   nearVal=0.01, farVal=100)
        pos, ori = [list(l) for l in
                    p.getBasePositionAndOrientation(car_id, client_id)]
        pos[2] = 0.2

        # Rotate camera direction
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = p.computeViewMatrix([pos[0], pos[1], pos[2]+7.5], pos + camera_vec, up_vec)

        # Display image
        frame = p.getCameraImage(400, 400, view_matrix, proj_matrix)[2]
        frame = np.reshape(frame, (400, 400, 4))
        self.rendered_img.set_data(frame)
        plt.draw()
        plt.pause(.00001)

    def close(self):
        p.disconnect(self.client)

    def calculate_distance(self, point1, point2):
        # 计算两个点之间的欧几里德距离
        distance = math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2 + (point2[2] - point1[2]) ** 2)
        return distance

    def find_coordinates_within_distance(self, target, coordinates):
        indices = []
        for i in range(len(coordinates)):
            distance = self.calculate_distance(target, coordinates[i])
            if distance < 1:
                indices.append(i)
        return indices


    def compute_car0_reward(self):
        self.car0_process_all_apples()
        self.car0_process_all_bombs()
        return self.car0_reward

    def car0_process_bomb(self, index):
        ball_name = "ball_" + str(index)
        if self.bomb_true[index] == 1:
            ball = getattr(self, ball_name)
            location_rock = ball.get_observation()
            location_car = self.car0.get_observation()
            distance = math.sqrt(
                (location_rock[0] - location_car[0]) ** 2 +
                (location_rock[1] - location_car[1]) ** 2 +
                (location_rock[2] - location_car[2]) ** 2
            )
            if distance < 1:
                self.bomb_true[index] = 0
                p.removeBody(ball.ball1, self.client)
                self.car0_reward -= 5
                print("car0_移除 ball:", ball_name)
        # else:
        #         print("ball 不存在:", ball_name)

    def car0_process_all_bombs(self):
        for i in range(len(self.bomb_true)):
            self.car0_process_bomb(i)
    def car0_process_apple(self, index):
        rock_name = "rock_" + str(index)
        if self.apple_true[index] == 1:
            rock = getattr(self, rock_name)
            location_rock = rock.get_observation()
            location_car = self.car0.get_observation()
            distance = math.sqrt(
                (location_rock[0] - location_car[0]) ** 2 +
                (location_rock[1] - location_car[1]) ** 2 +
                (location_rock[2] - location_car[2]) ** 2
            )
            if distance < 1:
                self.apple_true[index] = 0
                p.removeBody(rock.rock_1, self.client)
                self.car0_reward += 10
                print("car0_移除 rock:", rock_name)
        # else:
        #         print("ball 不存在:", ball_name)

    def car0_process_all_apples(self):
        for i in range(len(self.apple_true)):
            self.car0_process_apple(i)
    def compute_car1_reward(self):

        self.car1_process_all_apples()
        self.car1_process_all_bombs()
        return self.car1_reward
    def car1_process_bomb(self, index):
        ball_name = "ball_" + str(index)
        if self.bomb_true[index] == 1:
            ball = getattr(self, ball_name)
            location_rock = ball.get_observation()
            location_car = self.car1.get_observation()
            distance = math.sqrt(
                (location_rock[0] - location_car[0]) ** 2 +
                (location_rock[1] - location_car[1]) ** 2 +
                (location_rock[2] - location_car[2]) ** 2
            )
            if distance < 1:
                self.bomb_true[index] = 0
                p.removeBody(ball.ball1, self.client)
                self.car1_reward -= 5
                print("car1_移除 ball:", ball_name)
        # else:
        #         print("ball 不存在:", ball_name)

    def car1_process_all_bombs(self):
        for i in range(len(self.bomb_true)):
            self.car1_process_bomb(i)
    def car1_process_apple(self, index):
        rock_name = "rock_" + str(index)
        if self.apple_true[index] == 1:
            rock = getattr(self, rock_name)
            location_rock = rock.get_observation()
            location_car = self.car1.get_observation()
            distance = math.sqrt(
                (location_rock[0] - location_car[0]) ** 2 +
                (location_rock[1] - location_car[1]) ** 2 +
                (location_rock[2] - location_car[2]) ** 2
            )
            if distance < 1:
                self.apple_true[index] = 0
                p.removeBody(rock.rock_1, self.client)
                self.car1_reward += 10
                print("car1_移除 rock:", rock_name)
        # else:
        #         print("ball 不存在:", ball_name)

    def car1_process_all_apples(self):
        for i in range(len(self.apple_true)):
            self.car1_process_apple(i)
    def compute_car2_reward(self):

        self.car2_process_all_apples()
        self.car2_process_all_bombs()
        return self.car2_reward
    def car2_process_bomb(self, index):
        ball_name = "ball_" + str(index)
        if self.bomb_true[index] == 1:
            ball = getattr(self, ball_name)
            location_rock = ball.get_observation()
            location_car = self.car2.get_observation()
            distance = math.sqrt(
                (location_rock[0] - location_car[0]) ** 2 +
                (location_rock[1] - location_car[1]) ** 2 +
                (location_rock[2] - location_car[2]) ** 2
            )
            if distance < 1:
                self.bomb_true[index] = 0
                p.removeBody(ball.ball1, self.client)
                self.car2_reward -= 5
                print("car2_移除 ball:", ball_name)
        # else:
                # print("ball 不存在:", ball_name)

    def car2_process_all_bombs(self):
        for i in range(len(self.bomb_true)):
            self.car2_process_bomb(i)
    def car2_process_apple(self, index):
        rock_name = "rock_" + str(index)
        if self.apple_true[index] == 1:
            rock = getattr(self, rock_name)
            location_rock = rock.get_observation()
            location_car = self.car2.get_observation()
            distance = math.sqrt(
                (location_rock[0] - location_car[0]) ** 2 +
                (location_rock[1] - location_car[1]) ** 2 +
                (location_rock[2] - location_car[2]) ** 2
            )
            if distance < 1:
                self.apple_true[index] = 0
                p.removeBody(rock.rock_1, self.client)
                self.car2_reward += 10
                print("car2_移除 rock:", rock_name)
        # else:
        #         print("ball 不存在:", ball_name)

    def car2_process_all_apples(self):
        for i in range(len(self.apple_true)):
            self.car2_process_apple(i)