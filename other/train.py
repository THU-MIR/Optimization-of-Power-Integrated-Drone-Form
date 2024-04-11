import numpy as np
import leg_robot_gym 
import  gymnasium
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from ppo import PPO
from gymnasium.spaces import Box, Space

class VecExtractDictObs(VecEnvWrapper):

    def __init__(self, venv: VecEnv, key: str):
        self.key = key
        super().__init__(venv=venv, observation_space=venv.observation_space)

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, done, info = self.venv.step_wait()
        return obs, reward, done, info

#env = DummyVecEnv([lambda: gymnasium.make("servo_leg-v0",render_mode="human")])
# Wrap the VecEnv
#env = VecExtractDictObs(env, key="observation")
env=gymnasium.make("servo_leg-v0",render_mode="rgb_array")
env_test=gymnasium.make("servo_leg-v0",render_mode="human")
shape_observation_space = Box(low=-1, high=1, shape=(20,), dtype=np.float64)
shape_action_space = Box(low=-1, high=1, shape=(20,), dtype=np.float64)
model = PPO("MlpPolicy", env, verbose=1,n_steps=100,n_epochs= 10,stats_window_size=1000,shape_observation_space=shape_observation_space,shape_action_space=shape_action_space)
model.learn(1000)

vec_env = model.get_env()
obs,_=env_test.reset()
obs= vec_env.reset()
obs=obs.squeeze(axis=0)
for i in range(10000):
    action, _state = model.predict(obs, deterministic=True)
    shape_action,_s=model.shape_policy.predict(obs[70:],deterministic=True)
    action=np.concatenate((action,shape_action))
    obs, reward,done,_,info = env_test.step(action)
    if done:
        obs,_=env_test.reset()