

from gymnasium.envs.mujoco.mujoco_env import DEFAULT_SIZE
import numpy as np
import math
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import mujoco
from os import path
import torch
from robot_xml.make_bot import robot_transform
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
import random
from scipy.spatial.transform import Rotation as R


DEFAULT_CAMERA_CONFIG = {
    "distance": 30.0,
}

class servo_legEnv_v3(MujocoEnv,utils.EzPickle):
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
                 task:str="land",
                 is_transform:bool=False,
                 lock:bool=False,
                 reset_noise_scale=0.1,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.0, 5.0),
                 healthy_reward=5.0,
                 ctrl_cost_weight=0.00005
                 ):
        self.render_mode=render_mode
        xml_file="~/Desktop/m_robot/robot_xml/"+xml_file
        utils.EzPickle.__init__(self,
                                xml_file,
                                self.render_mode,
                                task,
                                reset_noise_scale,
                                terminate_when_unhealthy,
                                healthy_z_range,
                                healthy_reward,
                                ctrl_cost_weight
                                )
        self.obs_shape = 82


        self.shape_trans=np.zeros((12))
        self.reset_varible()
        self.stage='transform_stage'

        self.is_transform=is_transform
        self.is_lock_joint=lock

        self.task=task
        self.task_set()
        
    


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
        self.rendermode=render_mode
        self.default_camera_config=DEFAULT_CAMERA_CONFIG
        self.thrust=torch.zeros((4))
       

    def reset_varible(self):
        self.tag =True
        self.t=0
        self._t=0
        self.divide_t=0
        self._last_action=np.zeros((32))
        self.time_reward=0

    def string_mujo_init(self,xml_string):
       
        del self.model
        del self.data
        self.close()
        del self.mujoco_renderer
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        #self.model=xml_file
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height
        self.data = mujoco.MjData(self.model)
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        self._set_action_space()
        self._reset_simulation()
        self.mujoco_renderer=MujocoRenderer(self.model,self.data,self.default_camera_config)
        if self.render_mode == "human":
            self.render()
        self.tag=False

    def file_mujo_init(self,xml_file):

        xml_file=xml_file
        self.fullpath=path.expanduser(xml_file)
        self.mujoco_renderer.close()
        MujocoEnv.__init__(self, 
                           xml_file, 
                           5, 
                           observation_space=self._observation_space, 
                           default_camera_config=DEFAULT_CAMERA_CONFIG,
                           render_mode=self.render_mode
                           )
        if self.render_mode == "human":
            self.render()
        self.tag=False

    def head_rew(self,quat,target_position,xyz_position):
        '''angle=2*torch.acos(quat[0])
        xq=quat[1:]/torch.sqrt(1-quat[0]*quat[0])
        r=torch.sqrt(torch.square(xq).sum(-1))
        xq=xq/r
        head_v=target_position[:2]-xyz_position[:2]
        r=torch.sqrt(torch.square(target_position[:2] - xyz_position[:2]).sum(-1))
        head_v=head_v/r
        head_rew=1/(1+torch.sqrt(torch.square(head_v-xq[:2]).sum(-1)))'''
        target_quat=torch.zeros((4))
        head_v=target_position[:2]-xyz_position[:2]
        head_a=torch.acos(head_v[0]/torch.norm(head_v))
        target_quat[0]=torch.cos(head_a/2)
        target_quat[3]=1*torch.sin(head_a/2)
        head_rew=1/(1.0+torch.norm(target_quat-quat))
        return head_rew
    def quaternion2euler(quaternion):
        r = R.from_quat(quaternion)
        euler = r.as_euler('xyz', degrees=True)
        return euler
    
    def task_set(self):
        if self.task=="hover":
            self.target_position=np.zeros((3))
            self.target_position[2]=5
            #self.target_position[1]=1
            self.target=3
            self.obs_shape=73
            if self.is_transform:
                self.obs_shape=85
        if self.task == "speed":
            self.target_position=5
            self.target_vel=np.array([0.5,0,0])
            self.target=4
            self.obs_shape=86
        if self.task=="circle" or self.task=="8":
            self.target_position=np.zeros((3))
            self.target_position[2]=5
            self.target_position[0]=0
            self.target_position[1]=0
            self.target=3
            self.obs_shape=73
            if self.is_transform:
                self.obs_shape=85
        if self.task=="land":
            self.target_position=np.zeros((2))
            self.target_position[0]=5
            self.target_position[1]=0
            self.target=2
            self.obs_shape=72
            if self.is_transform:
                self.obs_shape=85

    def circle_task(self,dist,task):
        radius=6
        divide=0.5
        t=self.t-80
        self.time_reward=0
        if t<0 and self.divide_t ==0:
            self.target_position[0]=radius/2
        self._t+=1
        if dist < 1 :
            self.divide_t+=1
            a=math.radians(divide*self.divide_t)
            if task=="circle":
                self.target_position[0]=radius*math.cos(a)
                self.target_position[1]=radius*math.sin(a)
                self.time_reward=4/(1.0+self._t)
                self._t=0
            if task=="8":
                self.target_position[0]=radius*math.cos(a)/(1+math.sin(a)**2)
                self.target_position[1]=2*radius*math.sin(a)*math.cos(a)/(1+math.sin(a)**2)
                self.time_reward=4/(1.0+self._t)
                self._t=0
                

    @property
    def terminated(self):
        '''if self.position[2]<0:
            terminated=True
            return terminated'''
        
        
        #if self.quat[2] <0:# or self.quat[2]<-0.5:
            #terminated= True
            #return terminated
        if self.t>1000:
            terminated=True
            return terminated
        if self.target_dist> 15:
            terminated=True
            return terminated
        if self.position[2]>3:
            terminated=True
            return terminated
        else:
            terminated=False
        #terminated= True if self.t >1500 else False
        
    
    
    '''def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        low_shape=np.ones((12))
        high_shape=np.ones((12))
        
        #low_thrust=np.zeros((4))
        #high_thrust=np.ones((4))
        for i in range(12):
            low_shape[i]=-1
        low=np.append(low[28:32],low_shape)
        high=np.append(high[28:32],high_shape)
        #low=np.append(low_thrust,low_shape)
        #high=np.append(high_thrust,high_shape)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space'''
        
    
    def _get_obs(self):
        self.position=self.data.qpos.flat.copy()
        self.velocity=self.data.qvel.flat.copy()
        self.ctrl=self.data.ctrl.flat.copy()
        if self.stage=="transform_stage":
            tag=[0]
        elif self.stage=="execute_stage":
            tag=[1]
        if self.task=="hover":
            self.distance=self.position[0:3]-self.target_position
            if self.is_transform:
                return np.concatenate((self.position,self.velocity,self.shape_trans,self.distance,tag))
            else:
                return np.concatenate((self.position,self.velocity,self.distance,tag))
        if self.task=="speed":
            self.distance=np.array([self.position[2]-self.target_position])
            self.vel_dist=self.velocity[0:3]-self.target_vel
            if self.is_transform:
                return np.concatenate((self.position,self.velocity,self.shape_trans,self.distance,self.vel_dist,tag))
            else:
                return np.concatenate((self.position,self.velocity,self.distance,self.vel_dist,tag))
        if self.task=="circle" or self.task=="8":
            self.distance=self.position[0:3]-self.target_position
            if self.is_transform:
                return np.concatenate((self.position,self.velocity,self.shape_trans,self.distance,tag))
            else:
                return np.concatenate((self.position,self.velocity,self.distance,tag))
          
        if self.task=="land" :
            self.distance=self.position[0:2]-self.target_position
            if self.is_transform:
                return np.concatenate((self.position,self.velocity,self.shape_trans,self.distance,tag))
            else:
                return np.concatenate((self.position,self.velocity,self.distance,tag))
    
    def control_cost(self, action):
        con=action[:28]-self._last_action[:28]
        control_cost = self._ctrl_cost_weight * np.sum(np.square(con))
        self._last_action=action
        return control_cost
    
    def ctrl_range(self,actions,stage="execute_stage"):
        if stage =="execute_stage":
            low, high = self.action_space.low, self.action_space.high
            scale_actions=low + (0.5*(actions+1) * (high - low))
        if stage =="transform_stage":
            high=1.0
            low=-1.0
            scale_actions=low + (0.5*(actions+1)* (high - low))
        return scale_actions
    
    def step(self, actions):
        #actions=np.clip(actions,a_min=-1.0,a_max=1.0)
        if  not self.is_transform:
            self.stage ="execute_stage"
        if self.stage=="transform_stage":
            actions=self.ctrl_range(actions[:12],self.stage)
            qpos=self.data.qpos.flat.copy()
            qvel=self.data.qvel.flat.copy()
            info={
                "stage":self.stage
            }
            self.transform_robot(actions)
            self.set_state(qpos,qvel)
            observation=self._get_obs()
            reward=0.0
            return observation,reward,False,False,info
        #if self.tag :
        #   self.transform_robot(actions)
        elif self.stage=="execute_stage":
            '''self.last_actions+=actions.cpu().numpy()
            low, high = self.action_space.low, self.action_space.high
            self.last_actions=np.clip(self.last_actions,low,high)'''
            if self.is_lock_joint:
                if self.task=="circle" or self.task=="8": 
                    joint_action=np.zeros((28,))
                    joint_action[2]=10
                    joint_action[8]=-10
                    joint_action[13]=10
                    joint_action[16]=-10
                    actions=np.append(joint_action,actions[-4:])
                if self.task=="land":
                    joint_action=np.zeros((24,))
                    joint_action[3]=actions[3]
                    joint_action[9]=actions[9]
                    joint_action[19]=actions[19]
                    joint_action[22]=actions[22]
                    actions=np.append(joint_action,actions[-8:-4])
                    actions=np.append(actions,np.zeros((4)))
            #actions=self.ctrl_range(actions,self.stage)
            
            self.do_simulation(actions, self.frame_skip)

            self.quat=self.data.body("torso").xquat.copy()
            velocity=self.data.qvel.flat.copy()
            xyz_position = self.get_body_com("torso")[:3].copy()
            
            quat=torch.tensor(self.quat)
            velocity=torch.tensor(velocity)
            xyz_position=torch.tensor(xyz_position)
            control_rew=self.control_cost(actions)

            spin=velocity[3:6]
            spin=torch.sqrt(torch.square(torch.sum(spin)))
            spin_reward = torch.exp(-1.0 * spin)
            #up_reward=torch.exp(-1*torch.sum(torch.clamp(quat[3],min=0.0,max=1.0)))
            up_reward=torch.exp(-1*torch.sum(torch.clamp(quat[1:3],min=0.0,max=1.0)))
            
            '''xyz_velocity = (xyz_position_after - xyz_position_before) / self.dt
            x_velocity, y_velocity ,z_velocity= xyz_velocity
            forward_reward = xyz_position_after[0] - xyz_position_before[0]
            z_position=xyz_position_after[2]
            z_reward=math.exp(z_position-1)'''
            
            if self.task=="hover":

                target_position=torch.tensor(self.target_position)
                target_dist=torch.sqrt(torch.square(target_position - xyz_position).sum(-1))
                self.target_dist=target_dist
                pos_reward = 1.0 / (1.0 +target_dist)
                #pos_reward=pos_reward.clip(0,1)
                head_rew=self.head_rew(quat,torch.tensor((0,1,5)),xyz_position)
                rewards = pos_reward + pos_reward *(up_reward+spin_reward)
            if self.task=="circle" or self.task=="8":
                target_position=torch.tensor(self.target_position)
                target_dist=torch.sqrt(torch.square(target_position - xyz_position).sum(-1))
                self.target_dist=target_dist
                self.circle_task(target_dist,self.task)
                head_rew=self.head_rew(quat,target_position,xyz_position)
                pos_reward = 1.0 / (1.0 +target_dist)
                #pos_reward=pos_reward.clip(0,1)
                if self._t>20:
                    pos_reward=0
                rewards = pos_reward + pos_reward *(up_reward+spin_reward)+self.time_reward

            if self.task=="land" :
                target_position=torch.tensor(self.target_position)
                target_dist=torch.sqrt(torch.square(target_position - xyz_position[:2]).sum(-1))
                self.target_dist=target_dist
                #self.circle_task(target_dist,"circle")
                pos_reward = 1.0 / (1.0 +target_dist)
                #pos_reward=pos_reward.clip(0,1)
                rewards = pos_reward + pos_reward *(up_reward+spin_reward)
            
            elif self.task=="speed":
                target_vel=torch.tensor(self.target_vel)
                target_position=torch.tensor(self.target_position)
                target_dist=torch.sqrt(torch.square(target_position - xyz_position[2]).sum(-1))
                self.target_dist=target_dist
                pos_reward = 1.0 / (1.0 +target_dist)
                vel_dist=torch.abs(velocity[0]-target_vel).sum(-1)
                vel_reward=1.0 / (1.0 +vel_dist)
                if self._t>20:
                    pos_reward=0
                rewards = vel_reward + vel_reward *(up_reward+ spin_reward)+pos_reward+self.time_reward

            rewards=float(rewards.detach().cpu().numpy())

            
            observation = self._get_obs()
            terminated = self.terminated
            info = {
                "stage":self.stage,
                "x_position": xyz_position[0].cpu().numpy(),
                "y_position": xyz_position[1].cpu().numpy(),
                "z_position": xyz_position[2].cpu().numpy(),
                "x_target":self.target_position[0],
                "y_target":self.target_position[1],
                "distance_from_origin": np.linalg.norm(xyz_position.cpu(), ord=2),
                
            }
            reward = rewards

            if self.render_mode == "human":
                self.render()
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
        self.reset_varible()
        if self.is_transform:
            self.stage="transform_stage"
        else:
            self.stage="execute_stage"
        observation = self._get_obs()
        return observation
    
    def transform_robot(self,action):
        self.shape_trans=action
        self.robot.transform(p_args=self.shape_trans)
        xml_string=self.robot.to_xml_string()
        #xml_file=self.robot.save()
        self.string_mujo_init(xml_string)
        #self.file_mujo_init(xml_file)
        self.stage="execute_stage"

        
