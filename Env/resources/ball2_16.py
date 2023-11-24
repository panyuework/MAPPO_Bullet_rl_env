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

class ball2_16:
    def __init__(self, client):
        self.client = client
        f_name = os.path.join(os.getcwd(), 'Env/resources/apple_my.urdf')
        self.ball2_16 = p.loadURDF(fileName=f_name,
                   basePosition=[15.7, -12.8, 0],
                   physicsClientId=client, useFixedBase = True)

    def get_ids(self):
        return self.ball2_16
    def get_observation(self):
        # Get the position and orientation of the car
        pos, ang = p.getBasePositionAndOrientation(self.ball2_16, self.client)
        # ang = p.getEulerFromQuaternion(ang)
        # ori = (math.cos(ang[2]), math.sin(ang[2]))
        # #pos = pos[:2]
        # # Get the velocity of the car
        # vel = p.getBaseVelocity(self.x10car_v0, self.client)[0][0:2]
        #
        # # Concatenate position, orientation, velocity
        # observation = (pos + ori + vel)
        return pos

    def apply_action(self):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)

        # 获取球体的当前位置和姿态
        pos, orn = p.getBasePositionAndOrientation(self.ball2_16, physicsClientId=self.client)

        # 计算施加在球体上的力矩
        torque = [y, -x, 0]

        # 施加力矩来滚动球体
        p.applyExternalTorque(self.ball2_16, -1, torque, p.WORLD_FRAME)