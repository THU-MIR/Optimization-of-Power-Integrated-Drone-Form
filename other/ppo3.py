from argparse import Action
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from datetime import datetime
from elegantrl.agents import net

# ------------------------------------- #
# 策略网络--输出连续动作的高斯分布的均值和标准差
# ------------------------------------- #

class PolicyNet(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions,n_tr_states=1,n_tr_actions=12):
        super(PolicyNet, self).__init__()
        self.tr_mu=self.net(n_states,n_tr_actions,n_hiddens)
        self.mu=self.net(n_states,n_actions,n_hiddens)
        self.zero=torch.zeros((1,20)).to("cuda")
        self.one=torch.ones((1,20)).to("cuda")
        self.log_std = nn.Parameter(torch.ones(n_actions) * 0, requires_grad=True)
        self.tr_log_std = nn.Parameter(torch.ones(1, n_tr_actions) * 0, requires_grad=True)

    # 前向传播
    def forward(self, x):
        if x[0,-1]==0:
            tr_mu=self.tr_mu(x)
            #tr_std=self.tr_std(x)
            tr_std=self.tr_log_std.exp().expand_as(tr_mu)
            return tr_mu,tr_std
        elif x[0,-1]==1:
            mu=self.mu(x)
            std=self.log_std.exp().expand_as(mu)
            return mu, std
    
    def net(self,n_states,n_actions,n_hiddens):
        net_mu = nn.Sequential(
            nn.Linear(n_states, n_hiddens),  
            nn.ReLU(), 
            #nn.Linear(n_hiddens, n_hiddens),  
            #nn.ReLU(),  
            nn.Linear(n_hiddens, n_actions),  
            nn.Softmax() 
        )
        '''net_std=nn.Sequential(
            nn.Linear(n_states, n_hiddens),  
            nn.Tanh(),  
            nn.Linear(n_hiddens, n_hiddens), 
            nn.Tanh(), 
            nn.Linear(n_hiddens, n_actions), 
            nn.Softplus()  
        )'''
        return net_mu



# ------------------------------------- #
# 价值网络 -- 评估当前状态的价值
# ------------------------------------- #

class ValueNet(nn.Module):
    def __init__(self, n_states, n_hiddens):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_hiddens)
        self.fc3 = nn.Linear(n_hiddens, 1)
    # 前向传播
    def forward(self, x):  
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x =torch.relu(x)
        x = self.fc2(x)
        x=torch.relu(x) 
        x = self.fc3(x) # [b,n_hiddens]-->[b,1]
        return x
def tensorfy(np_list, device=torch.device('cpu')):
    if isinstance(np_list[0], list):
        return [[torch.tensor(x).to(device) if i < 2 else x for i, x in enumerate(y)] for y in np_list]
    else:
        return [torch.tensor(y).to(device) for y in np_list]
# ------------------------------------- #
# 模型构建--处理连续动作
# ------------------------------------- #

class PPO:
    def __init__(self, n_states, n_hiddens, n_actions,
                 actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        # 实例化策略网络
        self.actor = PolicyNet(n_states, n_hiddens, n_actions).to(device)
        # 实例化价值网络
        self.critic = ValueNet(n_states, n_hiddens).to(device)
        # 策略网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # 价值网络的优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
       
        # 属性分配
        self.lmbda = lmbda  # GAE优势函数的缩放因子
        self.epochs = epochs  # 一条序列的数据用来训练多少轮
        self.eps = eps  # 截断范围
        self.gamma = gamma  # 折扣系数
        self.device = device 
    
    # 动作选择
    def take_action(self, state,policy_dist='gaussian'):  # 输入当前时刻的状态
        action={}
        # [n_states]-->[1,n_states]-->tensor
        state = torch.tensor(state[np.newaxis,:],dtype=torch.float).to(self.device)
        # 预测当前状态的动作，输出动作概率的高斯分布
        mu, std = self.actor(state)

        # 构造高斯分布
        if policy_dist == 'gaussian':
            execute_dict = torch.distributions.Normal(mu.view(-1), std.view(-1))
            execute_action = execute_dict.sample()
            action=execute_action
        if policy_dist == 'beta':
            execute_dict=torch.distributions.beta.Beta(mu.view(-1),std.view(-1))
            execute_action = execute_dict.sample()
            action=execute_action
        else :
            action=mu.view(-1)
        if len(action)<32:
            zeros=torch.zeros((32-len(action))).to("cuda")
            action=torch.cat((action,zeros))
        return action  # 返回动作组

    # 训练
    def update(self, transition_dict,is_transform=False):
        # 提取数据集
        states = torch.tensor(transition_dict.states, dtype=torch.float).to(self.device)
        actions =torch.tensor(transition_dict.actions, dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict.rewards, dtype=torch.float).to(self.device).view(-1,1)  # [b,1]
        masks=torch.tensor(transition_dict.masks, dtype=torch.float).to(self.device).view(-1,1)
        next_states = torch.tensor(transition_dict.next_states, dtype=torch.float).to(self.device)  # [b,n_states]
        
        
        # 价值网络--目标，获取下一时刻的state_value  [b,n_states]-->[b,1]
        next_states_target = self.critic(next_states)
        # 价值网络--目标，当前时刻的state_value  [b,1]
        td_target = rewards + self.gamma * next_states_target * masks
        # 价值网络--预测，当前时刻的state_value  [b,n_states]-->[b,1]
        td_value = self.critic(states)
        # 时序差分，预测值-目标值  # [b,1]
        td_delta = td_value - td_target

        # 对时序差分结果计算GAE优势函数
        td_delta = td_delta.cpu().detach().numpy()  # [b,1]
        advantage_list = []  # 保存每个时刻的优势函数
        advantage = 0  # 优势函数初始值
        # 逆序遍历时序差分结果，把最后一时刻的放前面
        for delta in td_delta[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        # 正序排列优势函数
        advantage_list.reverse()
        # numpy --> tensor
        advantage = torch.tensor(advantage_list, dtype=torch.float).to(self.device)
        if is_transform:
            states_1=states[0,:].view(1,-1)
            states_2=states[1:,:]
            actions_1=actions[0,:12]
            actions_2=actions[1:,:4]
            # 策略网络--预测，当前状态选择的动作的高斯分布
            tr_mu, tr_std,= self.actor(states_1)  # [b,1]
            mu, std,= self.actor(states_2) 
            # 基于均值和标准差构造正态分布
            action_dists_1 = torch.distributions.Normal(tr_mu, tr_std)
            action_dists_2 = torch.distributions.Normal(mu, std)
      
        # 从正态分布中选择动作，并使用log函数
            tr_old_log_prob = action_dists_1.log_prob(actions_1).detach()
            old_log_prob = action_dists_2.log_prob(actions_2).detach()
        else:
            mu, std,= self.actor(states) 
            # 基于均值和标准差构造正态分布
            action_dists = torch.distributions.Normal(mu, std)
            old_log_prob = action_dists.log_prob(actions[:,:4]).detach()
        # 一个序列训练epochs次
        for _ in range(self.epochs):
            # 策略网络--预测，当前状态选择的动作的高斯分布
            if is_transform:
                tr_mu, tr_std,= self.actor(states_1)  # [b,1]
                mu, std,= self.actor(states_2) 
                # 基于均值和标准差构造正态分布
                action_dists_1 = torch.distributions.Normal(tr_mu, tr_std)
                action_dists_2 = torch.distributions.Normal(mu, std)
            
                # 从正态分布中选择动作，并使用log函数
                tr_log_prob = action_dists_1.log_prob(actions_1)
                log_prob = action_dists_2.log_prob(actions_2)
                self.backward(tr_log_prob,tr_old_log_prob,advantage[0,:],states_1,td_target[0,:])
                self.backward(log_prob,old_log_prob,advantage[1:,:],states_2,td_target[1:,:])
            else:
                mu, std,= self.actor(states) 
                action_dists = torch.distributions.Normal(mu, std)
                log_prob = action_dists.log_prob(actions[:,:4])
                self.backward(log_prob,old_log_prob,advantage,states,td_target)

    def backward(self,log_prob,old_log_prob,advantage,states,td_target):
            ratio = torch.exp(log_prob - old_log_prob)
            # 公式的左侧项
            surr1 = ratio * advantage
            # 公式的右侧项，截断
            surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps)
            # 策略网络的损失PPO-clip
            actor_loss = torch.mean(-torch.min(surr1,surr2))
            # 价值网络的当前时刻预测值，与目标价值网络当前时刻的state_value之差
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            # 优化器清0
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            # 梯度反传
            actor_loss.backward()
            critic_loss.backward()
            #梯度裁剪
            '''nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)  '''
            # 参数更新
            self.actor_optimizer.step()
            self.critic_optimizer.step()
    
    def save(self):
        time=str(datetime.now()).replace(" ","_").split(".")
        torch.save(self.actor,'policy1_'+time[0]+time[1]+'.pth')
        torch.save(self.critic,'critic1_'+time[0]+time[1]+'.pth')
    
    