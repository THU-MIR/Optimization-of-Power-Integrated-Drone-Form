

from gymnasium.envs.mujoco.mujoco_env import DEFAULT_SIZE
import numpy as np
import math
from gymnasium import spaces
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box, Space
import xml
import mujoco
from os import path
import torch
from robot_xml.make_bot import robot_transform
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
import random



DEFAULT_CAMERA_CONFIG = {
    "distance": 20.0,
}

class servo_legEnv_v1(MujocoEnv,utils.EzPickle):
    '''
    | Num | 观察空间                            | Min    | Max    | xml文件中的名称                          | 关节类型 | 单位            |
    |-----|一----------------------------------|--------|--------|----------------------------------------|---------|----------------|
    | 0   | x-躯干坐标                          | -Inf   | Inf    | torso                                  | free  | 距离(m)           |qpos[35]
    | 1   | y-躯干坐标                          | -Inf   | Inf    | torso                                  | free  | 距离(m)           |
    | 2   | z-躯干坐标                          | -Inf   | Inf    | torso                                  | free  | 距离(m)           |
    | 3   | x-躯干方向                          | -Inf   | Inf    | torso                                  | free  | 角度              |
    | 4   | y-躯干方向                          | -Inf   | Inf    | torso                                  | free  | 角度              |
    | 5   | z-躯干方向                          | -Inf   | Inf    | torso                                  | free  | 角度              |
    | 6   | w-躯干方向                          | -Inf   | Inf    | torso                                  | free  | 角度              |
    | 7   | 左前第一关节x轴                      | -90    | 90     | lf_leg1_Joint_x                        | hinge | 角度              |
    | 8   | 左前第一关节z轴                      | -90    | 90     | lf_leg1_Joint_z                        | hinge | 角度              |
    | 9   | 左前第二关节x轴                      | -90    | 90     | lf_leg2_Joint_x                        | hinge | 角度              |
    | 10  | 左前第二关节z轴                      | -90    | 90     | lf_leg2_Joint_z                        | hinge | 角度              |
    | 11  | 左前第三关节x轴                      | -90    | 90     | lf_leg3_Joint_x                        | hinge | 角度              |
    | 12  | 左前第三关节z轴                      | -90    | Inf    | lf_leg3_Joint_z                        | hinge | 角度              |
    | 13  | 左前驱动                            | -Inf   | Inf    | lf_motor_y                             | hinge | 角度              |
    | 14  | 右前第一关节x轴                      | -90    | 90     | rf_leg1_Joint_x                        | hinge | 角度              |
    | 15  | 右前第一关节z轴                      | -90    | 90     | rf_leg1_Joint_z                        | hinge | 角度              |
    | 16  | 右前第二关节x轴                      | -90    | 90     | rf_leg2_Joint_x                        | hinge | 角度              |
    | 17  | 右前第二关节z轴                      | -90    | 90     | rf_leg2_Joint_z                        | hinge | 角度              |
    | 18  | 右前第三关节x轴                      | -90    | 90     | rf_leg3_Joint_x                        | hinge | 角度              |
    | 19  | 右前第三关节z轴                      | -90    | 90     | rf_leg3_Joint_z                        | hinge | 角度              |
    | 20  | 右前驱动                            | -Inf   | Inf    | rf_motor_y                             | hinge | 角度              |
    | 21  | 左后第一关节x轴                      | -90    | 90     | lb_leg1_Joint_x                        | hinge | 角度              |
    | 22  | 左后第一关节z轴                      | -90    | 90     | lb_leg1_Joint_z                        | hinge | 角度              |
    | 23  | 左后第二关节x轴                      | -90    | 90     | lb_leg2_Joint_x                        | hinge | 角度              |
    | 24  | 左后第二关节z轴                      | -90    | 90     | lb_leg2_Joint_z                        | hinge | 角度              |
    | 25  | 左后第三关节x轴                      | -90    | 90     | lb_leg3_Joint_x                        | hinge | 角度              |
    | 26  | 左后第三关节z轴                      | -90    | 90     | lb_leg3_Joint_z                        | hinge | 角度              |
    | 27  | 左后驱动                            | -Inf   | Inf    | lb_motor_y                             | hinge | 角度              |
    | 28  | 右后第一关节x轴                      | -90    | 90     | rb_leg1_Joint_x                        | hinge | 角度              |
    | 29  | 右后第一关节z轴                      | -90    | 90     | rb_leg1_Joint_z                        | hinge | 角度              |
    | 30  | 右后第二关节x轴                      | -90    | 90     | rb_leg2_Joint_x                        | hinge | 角度              |
    | 31  | 右后第二关节z轴                      | -90    | 90     | rb_leg2_Joint_z                        | hinge | 角度              |
    | 32  | 右后第三关节x轴                      | -90    | 90     | rb_leg3_Joint_x                        | hinge | 角度              |
    | 33  | 右后第三关节z轴                      | -90    | 90     | rb_leg3_Joint_z                        | hinge | 角度              |
    | 34  | 右后驱动                            | -Inf   | Inf    | rb_motor_y                             |       | 角度              |
    | 35  | x-躯干速度                          | -Inf   | Inf    | torso                                  | free  | m/s              |qvel[34]
    | 36  | y-躯干速度                          | -Inf   | Inf    | torso                                  | free  | m/s              |
    | 37  | z-躯干速度                          | -Inf   | Inf    | torso                                  | free  | m/s              |
    | 38  | x-躯干角速度                        | -Inf   | Inf    | torso                                  | free  | °/s               |
    | 39  | y-躯干角速度                        | -Inf   | Inf    | torso                                  | free  | °/s               |
    | 40  | z-躯干角速度                        | -Inf   | Inf    | torso                                  | free  | °/s               |
    | 41  | 左前第一关节x轴角速度                 | -Inf   | Inf    | lf_leg1_Joint_x                        | hinge | °/s              |
    | 42  | 左前第一关节z轴角速度                 | -Inf   | Inf    | lf_leg1_Joint_z                        | hinge | °/s              |
    | 43  | 左前第二关节x轴角速度                 | -Inf   | Inf    | lf_leg2_Joint_x                        | hinge | °/s              |
    | 44  | 左前第二关节z轴角速度                 | -Inf   | Inf    | lf_leg2_Joint_z                        | hinge | °/s              |
    | 45  | 左前第三关节x轴角速度                 | -Inf   | Inf    | lf_leg3_Joint_x                        | hinge | °/s              |
    | 46  | 左前第三关节z轴角速度                 | -Inf   | Inf    | lf_leg3_Joint_z                        | hinge | °/s              |
    | 47  | 左前驱动角速度                       | -Inf   | Inf    | lf_motor_y                             | hinge | °/s              |
    | 48  | 右前第一关节x轴角速度                 | -Inf   | Inf    | rf_leg1_Joint_x                        | hinge | °/s              |
    | 49  | 右前第一关节z轴角速度                 | -Inf   | Inf    | rf_leg1_Joint_z                        | hinge | °/s              |
    | 50  | 右前第二关节x轴角速度                 | -Inf   | Inf    | rf_leg2_Joint_x                        | hinge | °/s              |
    | 51  | 右前第二关节z轴角速度                 | -Inf   | Inf    | rf_leg2_Joint_z                        | hinge | °/s              |
    | 52  | 右前第三关节x轴角速度                 | -Inf   | Inf    | rf_leg3_Joint_x                        | hinge | °/s              |
    | 53  | 右前第三关节z轴角速度                 | -Inf   | Inf    | rf_leg3_Joint_z                        | hinge | °/s              |
    | 54  | 右前驱动角速度                       | -Inf   | Inf    | rf_motor_y                             | hinge | °/s              |
    | 55  | 左后第一关节x轴角速度                 | -Inf   | Inf    | lb_leg1_Joint_x                        | hinge | °/s              |
    | 56  | 左后第一关节z轴角速度                 | -Inf   | Inf    | lb_leg1_Joint_z                        | hinge | °/s              |
    | 57  | 左后第二关节x轴角速度                 | -Inf   | Inf    | lb_leg2_Joint_x                        | hinge | °/s              |
    | 58  | 左后第二关节z轴角速度                 | -Inf   | Inf    | lb_leg2_Joint_z                        | hinge | °/s              |
    | 59  | 左后第三关节x轴角速度                 | -Inf   | Inf    | lb_leg3_Joint_x                        | hinge | °/s              |
    | 60  | 左后第三关节z轴角速度                 | -Inf   | Inf    | lb_leg3_Joint_z                        | hinge | °/s              |
    | 61  | 左后驱动角速度                       | -Inf   | Inf    | lb_motor_y                             | hinge | °/s              |
    | 62  | 右后第一关节x轴角速度                 | -Inf   | Inf    | rb_leg1_Joint_x                        | hinge | °/s              |
    | 63  | 右后第一关节z轴角速度                 | -Inf   | Inf    | rb_leg1_Joint_z                        | hinge | °/s              |
    | 64  | 右后第二关节x轴角速度                 | -Inf   | Inf    | rb_leg2_Joint_x                        | hinge | °/s              |
    | 65  | 右后第二关节z轴角速度                 | -Inf   | Inf    | rb_leg2_Joint_z                        | hinge | °/s              |
    | 66  | 右后第三关节x轴角速度                 | -Inf   | Inf    | rb_leg3_Joint_x                        | hinge | °/s              |
    | 67  | 右后第三关节z轴角速度                 | -Inf   | Inf    | rb_leg3_Joint_z                        | hinge | °/s              |
    | 68  | 右后驱动角速度                       | -Inf   | Inf    | rb_motor_y                             |       | °/s              |

    | 69  | 目标z间距                           | -Inf   | Inf    |                                        |       | 距离(m)           |


    | 35  | 目标x间距                           | -Inf   | Inf    |                                        |       | 距离(m)           |
    | 36  | 目标y间距                           | -Inf   | Inf    |                                        |       | 距离(m)           |
    | 37  | 目标z间距                           | -Inf   | Inf    |                                        |       | 距离(m)           |
    | 38  | 目标x速度                            | -Inf   | Inf    |                                        |       | m/s              |
    | 39  | 目标速度                            | -Inf   | Inf    |                                        |       | m/s              |
    
'''
    
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }
    def __init__(self, 
                 xml_file:str="lun_robot.xml",
                 render_mode ="rgb_array",
                 task_n:int=0,
                 reset_noise_scale=0.1,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.0, 5.0),
                 healthy_reward=5.0,
                 ctrl_cost_weight=0.05
                 ):
        self.render_mode=render_mode
        xml_file="~/Desktop/m_robot/robot_xml/"+xml_file
        utils.EzPickle.__init__(self,
                                xml_file,
                                self.render_mode,
                                task_n,
                                reset_noise_scale,
                                terminate_when_unhealthy,
                                healthy_z_range,
                                healthy_reward,
                                ctrl_cost_weight
                                )
        self.obs_shape = 113
        self.task={
            0:self.hover_task
        }
        self.task[task_n]()
        self.target=np.zeros(self.target)


        self._observation_space = Box(low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float64)
        self.robot=robot_transform()
    
        MujocoEnv.__init__(self, 
                           xml_file, 
                           5, 
                           observation_space=self._observation_space, 
                           default_camera_config=DEFAULT_CAMERA_CONFIG,
                           render_mode=self.render_mode
                           )
        
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._reset_noise_scale = reset_noise_scale
        self._healthy_z_range=healthy_z_range
        self._healthy_reward=healthy_reward
        self._ctrl_cost_weight=ctrl_cost_weight
        self._last_action=np.zeros((32))
        self.thrust=np.zeros((4))

        self.rendermode=render_mode
        self.default_camera_config=DEFAULT_CAMERA_CONFIG

        self.z_target_position=3
        self.shape_trans=np.zeros((12))
        self.tag=True
        self.task={
            "hover":self.hover_task()
        }

    def mujo_init(self,xml_file):

        xml_file="~/Desktop/m_robot/robot_xml/make/"+xml_file
        self.fullpath=path.expanduser(xml_file)
        self.model = mujoco.MjModel.from_xml_path(self.fullpath)
        #self.model=xml_file
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height
        self.data = mujoco.MjData(self.model)
        self._reset_simulation()

        self.mujoco_renderer.close()
        self.mujoco_renderer=MujocoRenderer(self.model,self.data,self.default_camera_config)
        if self.render_mode == "human":
            self.render()
        self.tag=False

    def quat_axis(slef,quat):
        angle=2*torch.acos(quat[0])
        x=quat[1]/torch.sqrt(1-quat[0]*quat[0])
        y=quat[2]/torch.sqrt(1-quat[0]*quat[0])
        z=quat[3]/torch.sqrt(1-quat[0]*quat[0])
        return torch.tensor([angle,x,y,z])
    
    def hover_task(self):
        self.target_position=np.zeros((3))
        self.target_position[2]=5
        self.target_position[1]=1
        self.target=3
        self.obs_shape+=3

    
    @property
    def terminated(self):
        terminated= True if self.target_dist> 8 else False
        #terminated=False
        return terminated
    
    
    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        low_shape=np.ones((12))
        high_shape=np.ones((12))
        for i in range(12):
            low_shape[i]=-1
        low=np.append(low,low_shape)
        high=np.append(high,high_shape)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space
        
    
    def _get_obs(self):
        self.position=self.data.qpos.flat.copy()
        self.velocity=self.data.qvel.flat.copy()
        self.ctrl=self.data.ctrl.flat.copy()
        self.distance=self.position[0:3]-self.target_position
        return np.concatenate((self.position,self.velocity,self.ctrl,self.shape_trans,self.distance))
    
    def control_cost(self, action):
        con=action-self._last_action
        control_cost = self._ctrl_cost_weight * np.sum(np.square(con))
        self._last_action=action
        return control_cost
    
    def step(self, actions):
        if self.tag :
           self.transform_robot(actions)
        self.actions=actions
        action=actions[:32]
        

        self.do_simulation(action, self.frame_skip)

        quat=self.data.body("torso").xquat.copy()
        velocity=self.data.qvel.flat.copy()
        xyz_position_after = self.get_body_com("torso")[:3].copy()
        
        quat=torch.tensor(quat).cuda(0)
        velocity=torch.tensor(velocity).cuda(0)
        xyz_position_after=torch.tensor(xyz_position_after).cuda(0)
        target_position=torch.tensor(self.target_position).cuda(0)


        spin=velocity[3:6]
        spinnage = torch.abs(velocity[..., 2])
        spinnage_reward = 1.0 / (1.0 + 0.1 * spinnage * spinnage)
        spin=torch.sum(spin)**2
        spin_reward = torch.exp(-1.0 * spin)
       
        axis=self.quat_axis(quat)
        up_reward=torch.exp(-1*torch.sum(torch.clamp(quat[1:3],min=0.0,max=1.0)))
        

        
        target_dist=torch.sqrt(torch.square(target_position - xyz_position_after).sum(-1))
        #target_dist = np.inner(target_dist,target_dist)
        self.target_dist=target_dist
        pos_reward = 1.0 / (1.0 +target_dist)  #
        
        
        '''xyz_velocity = (xyz_position_after - xyz_position_before) / self.dt
        x_velocity, y_velocity ,z_velocity= xyz_velocity
        forward_reward = xyz_position_after[0] - xyz_position_before[0]
        z_position=xyz_position_after[2]
        z_reward=math.exp(z_position-1)'''
        

        rewards = 2*pos_reward + pos_reward *(up_reward+ spin_reward)
        rewards=float(rewards.detach().cpu().numpy())

        costs= self.control_cost(action)
        observation = self._get_obs()
        terminated = self.terminated
        info = {
            "tag":self.tag,

            "x_position": xyz_position_after[0],
            "y_position": xyz_position_after[1],
            "z_position": xyz_position_after[2],
            "distance_from_origin": np.linalg.norm(xyz_position_after.cpu(), ord=2),
            
        }
        reward = rewards

        if self.render_mode == "human":
            self.render()
        if terminated :
            self.tag =True
        self.t+=1
        return observation, reward, terminated, False, info
    
    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        #self.target_position=np.random.randint(low=2,high=5,size=(3),dtype="int")
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        self.tag =True
        self.t=0

        return observation
    
    def transform_robot(self,action):
        self.shape_trans=action[32:]
        self.robot.transform(p_args=self.shape_trans)
        xml_file=self.robot.save()
        self.mujo_init(xml_file)


        
