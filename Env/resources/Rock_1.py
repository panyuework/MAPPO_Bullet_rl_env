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
import random

class rock_1:
    def __init__(self, client, location):
        self.client = client
        f_name = os.path.join(os.getcwd(), 'Env/resources/apple.urdf')
        self.rock_1 = p.loadURDF(fileName=f_name, basePosition=location,
                           # baseOrientation=p.getQuaternionFromEuler([0, 0, orientation]),
                           physicsClientId=self.client, useFixedBase = True)
        # self.rock_1 = self.generate_apple()

    def get_ids(self):
        return self.rock_1

    def get_observation(self):
        # Get the position and orientation of the car
        pos, ang = p.getBasePositionAndOrientation(self.rock_1, self.client)
        # ang = p.getEulerFromQuaternion(ang)
        # ori = (math.cos(ang[2]), math.sin(ang[2]))
        # #pos = pos[:2]
        # # Get the velocity of the car
        # vel = p.getBaseVelocity(self.x10car_v0, self.client)[0][0:2]
        #
        # # Concatenate position, orientation, velocity
        # observation = (pos + ori + vel)

        return pos
    def generate_apple(self):
        f_name = os.path.join(os.getcwd(), 'Env/resources/apple.urdf')  # 小车模型的文件名
        x = random.uniform(-9.5, 0)  # 随机生成 x 坐标
        y = random.uniform(-9.5, 0)  # 随机生成 y 坐标
        # orientation = random.uniform(0, 2 * math.pi)  # 随机生成朝向角度（弧度制）

        carId = p.loadURDF(fileName=f_name, basePosition=[x, y, 0],
                           # baseOrientation=p.getQuaternionFromEuler([0, 0, orientation]),
                           physicsClientId=self.client, useFixedBase = True)
        return carId