from matplotlib import pyplot as plt
from ppo import PPO
from torch import nn
import torch
import random
import gymnasium
import random
import leg_robot_gym 
from datetime import datetime
import multiprocessing
import numpy as np
class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a tuple."""
        self.memory.append([*args])

    def sample(self, min_batch_size=None):
        if min_batch_size is None:
            return self.memory
        else:
            if min_batch_size > len(self.memory):
                random_batch = random.sample(self.memory, len(self.memory))
            else:
                batch_size=min_batch_size*(len(self.memory)//min_batch_size)
                random_batch = random.sample(self.memory, batch_size)
            return random_batch

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)
    
class TrajBatch:

    def __init__(self, memory):
        self.batch = zip(*memory)
        self.states = np.stack(next(self.batch))
        self.actions = np.stack(next(self.batch))
        self.masks = np.stack(next(self.batch))
        self.next_states = np.stack(next(self.batch))
        self.rewards = np.stack(next(self.batch))
        

def tensorfy(np_list, device=torch.device('cpu')):
    if isinstance(np_list[0], list):
        return [[torch.tensor(x).to(device) if i < 2 else x for i, x in enumerate(y)] for y in np_list]
    else:
        return [torch.tensor(y).to(device) for y in np_list]

class agent():
    def __init__(self,env_name) -> None:
        self.evol_num=10#进化次数
        self.env_num=8#环境数量
        self.num_episodes = 2001  # 总迭代次数
        self.epoch=1#迭代几次更新一次
        self.gamma = 0.9  # 折扣因子
        self.actor_lr = 1e-4  # 策略网络的学习率
        self.critic_lr = 1e-4  # 价值网络的学习率
        
        self.n_hiddens = 256  # 隐含层神经元个数
        self.device = torch.device('cuda') if torch.cuda.is_available() \
                            else torch.device('cpu')
        self.env_name = env_name
        self.return_list = []  # 保存每个回合的return
        self.m_transition_dict=[]
        self.batch_size=1024
        self.traj_cls=TrajBatch
        
        self.n_obs,self.n_actions=self.set_env(env_name)
        self.n_actions=4
        self.set_policy(self.n_obs,self.n_actions,self.n_hiddens,self.actor_lr,self.critic_lr,self.gamma,self.device)

    def set_env(self,env_name,mode="human",task="hover"):
        self.env=gymnasium.make(env_name,render_mode=mode)
        if mode =="rgb_array":
            self.env=gymnasium.wrappers.RecordVideo(self.env,"./videos",lambda ep:ep%200==0)
        n_obs=self.env.observation_space.shape[0]
        n_actions=self.env.action_space.shape[0] 
        return n_obs,n_actions
    
    def set_policy(self,n_states,n_actions,n_hiddens,actor_lr,critic_lr,gamma,device):
        self.policy=PPO(n_states=n_states,  # 状态数
                                  n_hiddens=n_hiddens,  # 隐含层数
                                  n_actions=n_actions,  # 动作数
                                  actor_lr=actor_lr,  # 策略网络学习率
                                  critic_lr=critic_lr,  # 价值网络学习率
                                  lmbda = 0.9,  # 优势函数的缩放因子
                                  epochs = 5,  # 一组序列训练的轮次
                                  eps = 0.2,  # PPO中截断范围的参数
                                  gamma=gamma,  # 折扣因子
                                  device = device)
    
    def train(self):
        memory = Memory()
        tr=[]
        x=[]
        tr_mean=[]
        plt.ion()
        for ep in range(self.num_episodes):
            t=0
            states,_=self.env.reset()
            shape_states=states
            shape_transform=self.policy.take_action(shape_states)
            shape_transform=torch.ones_like(shape_transform)
            states,rewards,dones,_,info=self.env.step(shape_transform.detach().cpu().numpy())
            memory.push(shape_states,shape_transform.detach().cpu(),1,states,rewards)
            total_rewards=0
            while not dones:
                actions=self.policy.take_action(states)
                actions[0]=0.3
                actions[1]=0.3
                actions[2]=0.3
                actions[3]=0.3
                next_states,rewards,dones,_,info=self.env.step(actions.detach().cpu().numpy())
                mask = 0 if dones else 1
                memory.push(states,actions.detach().cpu(),mask,next_states,rewards)
                states=next_states
                total_rewards+=rewards
                if dones :
                    break
            tr.append(total_rewards)

            plt.clf()
            plt.plot(range(len(tr)),tr)
            if ep%10==0:
                tr_mean.append(np.sum(tr[-10:])/10)
                x.append((len(tr_mean)-1)*10)
            plt.plot(x,tr_mean)
            plt.pause(0.001)
            self.policy.update(self.traj_cls(memory.sample(min_batch_size=16)))
            memory.memory.clear()
        time=str(datetime.now()).replace(" ","_").split(".")
        name='reward'+"_"+time[0]+time[1]+".png"
        plt.savefig(name)
        self.policy.save()
        torch.save(self.policy,'policy_'+time[0]+time[1]+'.pth')
    def record(self,num):
        if num==100:
            return True
        else:
            return False
if __name__=="__main__":
    env_name = 'servo_leg-v3'
    a=agent(env_name)
    a.train()