import numpy as np
#TODO 离散/视觉输入
from Env.envs.env_20 import X10Car_Env_out20
#TODO 连续/向量输入
#from Env.envs.x10car_env_out20 import X10Car_Env_out20
env = X10Car_Env_out20()

class EnvCore(object):
   
    

    def __init__(self):
        self.agent_num = 3 
        self.obs_dim = [100, 100, 5]  
        self.action_dim = 9  

    def reset(self):
        
        obs = env.reset()
        # sub_agent_obs = []
        # for i in range(self.agent_num):
        #     sub_obs = obs
        #     sub_agent_obs.append(sub_obs)
        return obs
        # return sub_agent_obs

    def step(self, actions):
        
        # print("#####################################################donzuo########################333")
        # print(actions)
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        obs, reward, done, info = env.step(actions)
        for i in range(self.agent_num):
            # sub_agent_obs.append(obs)
            sub_agent_reward.append([reward[i]])
            sub_agent_done.append(done)
            sub_agent_info.append(info)

        return [obs, sub_agent_reward, sub_agent_done, sub_agent_info]
