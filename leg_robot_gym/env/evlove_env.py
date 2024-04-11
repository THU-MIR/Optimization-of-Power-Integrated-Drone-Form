import gymnasium
import numpy as np
import math
from gymnasium import error, logger, spaces
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box, Space
import xml
from robot_xml.make_bot import robot_transform


class evlove_Env(MujocoEnv,utils.EzPickle):
    def __init__(self, name:str="servo_leg-v0",render_mode:str="rgb_array"):
        self.env=gymnasium.make(name,render_mode=render_mode)

    def _set_action_space(self):
        return self.env._set_action_space()