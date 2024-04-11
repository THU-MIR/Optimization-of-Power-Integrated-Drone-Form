from math import e
import random
import gymnasium
import leg_robot_gym 
from stable_baselines3 import PPO
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from transform_policy import Transform_Policy
import torch
from matplotlib import pyplot as plt
#env=make_vec_env("servo_leg-v2",n_envs=1)
env=gymnasium.make("servo_leg-v3",render_mode="human")#servo_leg-v0,servo_leg-v1

#model = PPO(Transform_Policy, env, verbose=1,)
model=PPO.load(path="servo_leg-v3_land_NOtransform_2024-04-11_16:50:53.zip",env=env)
vec_env = model.get_env()
epoch=1
for i in range(epoch):
    obs = vec_env.reset()
    dones =0
    x=[]
    x_t=[]
    y=[]
    y_t=[]
    z=[]
    while not dones:
        action,_ = model.predict(obs,deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        if info[0]["stage"]=="execute_stage":

            x.append(info[0]["x_position"])
            x_t.append(info[0]["x_target"])
            y.append(info[0]["y_position"])
            y_t.append(info[0]["y_target"])
            z.append(info[0]["z_position"])
        if dones:
            plt.plot(x_t,y_t)
            plt.plot(x,y)
            plt.show()
            break
        #env.render()