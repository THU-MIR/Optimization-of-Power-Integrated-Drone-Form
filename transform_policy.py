from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch as th
from torch import nn
import torch
import leg_robot_gym 
import multiprocessing
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy


class Transform_Network(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        first_layer_dim_pi: int = 128,
        last_layer_dim_pi: int = 128,
        first_layer_dim_vf: int = 128,
        last_layer_dim_vf: int = 128,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_transform_net = nn.Sequential(
            nn.Linear(feature_dim, first_layer_dim_pi), nn.ReLU(),
            nn.Linear(first_layer_dim_pi, last_layer_dim_pi)
        )
        self.policy_actor_net = nn.Sequential(
            nn.Linear(feature_dim, first_layer_dim_pi), nn.ReLU(),
            nn.Linear(first_layer_dim_pi, last_layer_dim_pi)
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, first_layer_dim_vf), nn.ReLU(),
            nn.Linear(first_layer_dim_pi, last_layer_dim_vf), nn.ReLU(),
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def multi_forward(self,features):
        if features[-1]==0:
                return self.policy_transform_net(features)
        else:
                return self.policy_actor_net(features)
        
    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        '''pool = multiprocessing.Pool()
        r = pool.map(self.multi_forward, features)'''
        rows,cols=features.shape
        trans=torch.tensor([]).cuda('cuda')
        actor=torch.tensor([]).cuda('cuda')
        order=[]
        r=torch.tensor([]).cuda('cuda')
        for i,feature in enumerate(features):
            if feature[-1]==0:
                trans=torch.cat((trans,feature.unsqueeze_(0)))
                order.append(0)
            else:
                actor=torch.cat((actor,feature.unsqueeze_(0)))
                order.append(1)
        if trans.shape[0] :
            trans=iter(self.policy_transform_net(trans))
        if actor.shape[0]:
            actor=iter(self.policy_actor_net(actor))
        for o in order:
            if o==0:
                r=torch.cat((r,next(trans).unsqueeze(0)))
            else:
                r=torch.cat((r,next(actor).unsqueeze(0)))
        '''for i  in range(rows):
            if features[i,-1]==0:
                r.append(self.policy_transform_net(features[i,:]))

            if features[i,-1]==1:
                r.append(self.policy_actor_net(features[i,:]))'''
        #r=torch.cat(r,dim=0).reshape(rows,-1)
        return r
        '''if features[0,-1]==0:
            return self.policy_transform_net(features)
         if features[0,-1]==1:
          return self.policy_actor_net(features)
'''
    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class Transform_Policy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = Transform_Network(self.features_dim)
#model = PPO(CustomActorCriticPolicy, "servo_leg-v3", verbose=1)
#model.learn(5000)