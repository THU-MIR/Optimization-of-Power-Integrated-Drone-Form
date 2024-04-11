
import gymnasium
import torch
import leg_robot_gym 
from stable_baselines3 import PPO
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image
from stable_baselines3.common.callbacks import CheckpointCallback
from transform_policy import Transform_Policy
from videos.videorecorder import VideoRecorderCallback
from datetime import datetime
n_envs=8
n_steps=500
iterations=300
save_num=10
env_name="servo_leg-v3"
total_steps=n_envs*n_steps*iterations
video_step=int(total_steps/(save_num*n_envs))
save_step=n_envs*n_steps*iterations/save_num
eval_log_dir = "./eval_logs/"
task="land"
transform=False
lock=False   

env_test=gymnasium.make(env_name,render_mode="human",task=task,is_transform=transform,lock=lock)

'''eval_callback = EvalCallback(eval_env, best_model_save_path=eval_log_dir,
                              log_path=eval_log_dir, eval_freq=max(500 // n_envs, 1),
                              n_eval_episodes=5, deterministic=True,
                              render=False)'''

#env=gymnasium.make("servo_leg-v2",render_mode="human")#servo_leg-v0,servo_leg-v1
checkpoint_callback = CheckpointCallback(
  save_freq=5000,
  save_path="./logs/",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

env=make_vec_env(env_name,n_envs=n_envs,)
video_recorder = VideoRecorderCallback(env_name, render_freq=video_step)
model = PPO(Transform_Policy, env,n_steps=n_steps,verbose=1,tensorboard_log="./tensorboard/servor-v3/")
#model=PPO.load(path="servo_leg-v3_8_transform_2024-04-03_17:46:35.zip",env=env)
model.learn(total_steps,progress_bar=True,callback=checkpoint_callback,)
time=str(datetime.now()).replace(" ","_").split(".")
if transform:
   name_str='_transform_'
else:
   name_str='_NOtransform_'
if lock:
   name_str+='lockJoint_'   
model.save(path="./weight/servo_leg-v3_"+task+name_str+time[0])
#ec_env = model.get_env()
obs,_ = env_test.reset()
epoch=1
for i in range(epoch):
  done=0
  rew=0
  obs,_ = env_test.reset()
  while not done:
      action, _state = model.predict(obs, deterministic=True)
      obs, reward,done,_,info = env_test.step(action)
      rew+=reward
      print(rew)
      if done:
         print(i)
         break