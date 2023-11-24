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

import pybullet as p
import os
import math, random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

class x10car_v0:
    def __init__(self, client):
        self.client = client
        f_name = os.path.join(os.getcwd(), 'Env/resources/x10car_v0.urdf')
        self.x10car_v0 = p.loadURDF(fileName=f_name,
                              basePosition=[0.0,0.0,0], #baseOrientation=p.getQuaternionFromEuler([0,0,-90*math.pi/180]),
                              #basePosition=[random.choice([-0.108268,10.0,-10.0,-18.0]), random.choice([0,-10.0,15.0]), 0],
                              physicsClientId=client)

        # Joint indices as found by p.getJointInfo()
        self.steering_joints = [4, 6]
        self.drive_joints = [2, 3, 5, 7]
        # Joint speed
        self.joint_speed = 0
        # Drag constants
        self.c_rolling = 0.2
        self.c_drag = 0.01
        # Throttle constant 
        self.c_throttle = 20
        self.rendered_img = None


    def get_ids(self):
        return self.x10car_v0, self.client

    def apply_action(self, action):
        # Expects action to be two dimensional
        throttle, steering_angle = action

        # Clip throttle and steering angle to reasonable values
        throttle = min(max(throttle, -1), 1)
        steering_angle = max(min(steering_angle, 0.36), -0.36)

        # Set the steering joint positions
        p.setJointMotorControlArray(self.x10car_v0, self.steering_joints,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[steering_angle] * 2,
                                    physicsClientId=self.client)

        # Calculate drag / mechanical resistance 
        friction = -self.joint_speed * (self.joint_speed * self.c_drag +
                                        self.c_rolling)
        acceleration = self.c_throttle * throttle + friction
        # Each time step is 1/240 of a second
        self.joint_speed = self.joint_speed + 1/30 * acceleration
        if self.joint_speed < 0:
            self.joint_speed = 0

        # Set the velocity of the wheel joints
        p.setJointMotorControlArray(
            bodyUniqueId=self.x10car_v0,
            jointIndices=self.drive_joints,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[self.joint_speed] * 4,
            forces=[1.2] * 4,
            physicsClientId=self.client)

    def get_observation(self):
        # Get the position and orientation of the car 
        pos, ang = p.getBasePositionAndOrientation(self.x10car_v0, self.client)
        ang = p.getEulerFromQuaternion(ang)
        ori = (math.cos(ang[2]), math.sin(ang[2]))
        #pos = pos[:2]
        # Get the velocity of the car
        vel = p.getBaseVelocity(self.x10car_v0, self.client)[0][0:2]

        # Concatenate position, orientation, velocity
        observation = (pos + ori + vel)

        return observation
    def get_cossion(self):
        key = 0
        P_min, P_max = p.getAABB(self.x10car_v0)
        id_tuple = p.getOverlappingObjects(P_min, P_max)
        if len(id_tuple) > 1:
            for ID, name in id_tuple:
                if ID == self.x10car_v0:
                    continue
                elif str(name).lower() in ["plane"]:
                    continue
                else:
                    print(f"Hit happened! Hit object ID: {ID}, Name: {name}")
                    key = 1
        return key


    def get_distance(self):
        rayFrom = []
        rayTo = []
        numRays = 100 # number of lidar points
        rayLen = 12 # maximum lidar range
        rayFoV = 0.5 * math.pi # lidar FoV
        distance = []
        distance_fix = 0.0
        
        rayHitColor = [1, 0, 0]
        rayMissColor = [0, 1, 0]
        
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        pos, ang = p.getBasePositionAndOrientation(self.x10car_v0, physicsClientId=self.client)
        angEuler = p.getEulerFromQuaternion(ang)
        
        # Define rayFrom and rayTo coordinates
        lidarPosition = [
            0.265*math.cos(angEuler[2]) + pos[0],
            0.265*math.sin(angEuler[2]) + pos[1],
            0.15 + pos[2]
        ]
        lidarOrientation = 0.25 * math.pi - angEuler[2]
   
        for i in range(numRays):
            rayFrom.append(lidarPosition)
            rayTo.append([
                rayLen * math.sin(lidarOrientation  + rayFoV * float(i) / numRays),
                rayLen * math.cos(lidarOrientation + rayFoV * float(i) / numRays), 0.15
            ])
        # # Remove previous debug items 移除
        # p.removeAllUserDebugItems()
        # Shoot lidar
        lidarResults = p.rayTestBatch(rayFrom, rayTo)

        for i in range(numRays):
            hitObjectUid = lidarResults[i][0]
            hitFraction = lidarResults[i][2]
            hitPosition = lidarResults[i][3]

            if (hitObjectUid == -1):
                distance.append(12)
                # p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor)
    
            else:
                distance_fix = math.sqrt((hitPosition[0] - rayFrom[i][0])**2 + (hitPosition[1] - rayFrom[i][1])**2)
                if (distance_fix == 0.0):
                    distance_fix = 12
                distance.append(distance_fix)
                # p.addUserDebugLine(rayFrom[i], hitPosition, rayHitColor)

        #print(hitObjectUid, distance)
        #min_dist_to_wall = min(i for i in distance if i != -1) #min(dist_to_wall) #
        #print(hitObjectUid, min_dist_to_wall, hitPosition)

        return distance

    # def get_camImg(self, car_id, client_id):
    #     if self.rendered_img is None:
    #         self.rendered_img = plt.imshow(np.zeros((100, 100, 4)))
    #
    #     # Base information
    #     #car_id, client_id = self.x10car_v0.get_ids()
    #     proj_matrix = p.computeProjectionMatrixFOV(fov=120, aspect=1,
    #                                                nearVal=0.01, farVal=100)
    #     pos, ori = [list(l) for l in
    #                 p.getBasePositionAndOrientation(car_id, client_id)]
    #     #pos[2] = 0.2
    #     ang = p.getEulerFromQuaternion(ori)
    #     # Rotate camera direction
    #     rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
    #     camera_vec = np.matmul(rot_mat, [1, 0, 0])
    #     up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
    #     view_matrix = p.computeViewMatrix([pos[0]+0.39*math.cos(ang[2]), pos[1]+0.39*math.sin(ang[2]), pos[2]+0.05], pos + camera_vec, up_vec)
    #
    #     # Display image
    #     camImg = p.getCameraImage(400, 400, view_matrix, proj_matrix)
    #     frame = camImg[2]
    #     frame = np.reshape(frame, (400, 400, 4))
    #     self.rendered_img.set_data(frame)
    #     #plt.draw()
    #     #path = os.path.join(os.getcwd(), 'CarEnv/human_detected.png')
    #     #plt.savefig(path)
    #     #print('Human detected and picture clicked')
    #     # cam_data = pd.DataFrame({'RGB array':[frame], 'Depth array':[camImg[3]], 'Segmentation Mask':[camImg[4]]})
    #     # return cam_data

    def get_camImg(self):
        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((100, 100, 4)))

        # Base information
        proj_matrix = p.computeProjectionMatrixFOV(fov=120, aspect=1,
                                                   nearVal=0.01, farVal=100)
        pos, ori = [list(l) for l in
                    p.getBasePositionAndOrientation(self.x10car_v0, self.client)]
        ang = p.getEulerFromQuaternion(ori)
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = p.computeViewMatrix(
            [pos[0] + 0.39 * math.cos(ang[2]), pos[1] + 0.39 * math.sin(ang[2]), pos[2] + 0.05], pos + camera_vec,
            up_vec)

        # Display image
        camImg = p.getCameraImage(100, 100, view_matrix, proj_matrix)
        frame = camImg[2]
        frame = np.reshape(frame, (100, 100, 4))
        # print(frame)
        self.rendered_img.set_data(frame)
        # plt.draw()
        # plt.imshow(frame, cmap='gray')
        # 自定义颜色映射
        # cmap = ListedColormap(['black', 'red', 'green',])  # 假设有四个不同的区域
        #
        # # 显示分割结果
        # plt.imshow(frame, cmap=cmap)
        # plt.colorbar()  # 添加颜色条
        # plt.show()
        # plt.show()
        # plt.pause(0.001)

        return frame

    def cul_reward(self, action, min_dist_to_wall, obs_car, obs_task):
        # Compute reward
        car_x, car_y, car_z = obs_car[:3]
        distance = math.sqrt((car_x - obs_task[0]) ** 2 + (car_y - obs_task[1]) ** 2 + (car_z - obs_task[2]) ** 2)
        A = 10.0  # 缩放因子
        center = 0.0  # 距离的中心值
        variance = 1.0  # 方差
        # print('#################distance##########################')
        # print(distance)
        # print('#################min_dist_to_wall##########################')
        # print(min_dist_to_wall)
        # print('#################action##########################')
        # print(action)

        # reward_distance = A * math.exp(-(distance - center) ** 2 / (2 * variance))
        reward_distance = 20.0/(distance+1)
        # reward = 0.005 * (min_dist_to_wall**2) + 5.0*((min(max(action[0], 0), 1))**2) - 2.0*((max(min(action[1], 0.36), -0.36))**2)
        # reward = 0.005 * (min_dist_to_wall ** 2)
        distance_start = math.sqrt((car_x - 0) ** 2 + (car_y - 0) ** 2 + (car_z - 0) ** 2)

        # reward = ((min(max(action[0], -1), 1))**2) - ((max(min(action[1], 0.36), -0.36))**2)  + reward_distance
        # print('#################reward##########################')
        # print(reward)
        reward = - reward_distance
        # if (min_dist_to_wall > 2.0 and min_dist_to_wall < 2.5):
        #     reward = reward + 2.0

        # if ((min(max(action[0], 0), 1)) < 0.1):
        #     reward = reward - 10

        if (3 < distance and distance < 5):
            reward = reward + 30
        if (2 < distance and distance < 3):
            reward = reward + 50
        if (1.5 < distance and distance < 2):
            reward = reward + 100


        # if (len(co) > 0):
        #     reward = reward - 0.5
        if (min_dist_to_wall < 1.0):
            reward = reward - 0.5

        # print('#################reward##########################')
        # print(reward)
        # key_cossion = self.get_cossion()
        # print(key_cossion)

        return reward, distance