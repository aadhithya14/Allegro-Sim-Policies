#Rewarder module for sinkhorn cosine

import numpy as np
import torch 
import sys
from openteach.robot.allegro.allegro import AllegroHand
from allegro_sim.utils import cosine_distance, optimal_transport_plan
from isaacgym import gymapi, gymutil
import gym
from isaacgym import gymtorch
from isaacgym.torch_utils import *

from .rewarder import Rewarder
import numpy as np

class ObjectDroppingPenaltySim(Rewarder):
    def __init__(
            self,
            
            exponential_weight_init=False, 
            **kwargs
    ):
        super().__init__(**kwargs)
        self.exponential_weight_init = exponential_weight_init
        
        self.fixed_height=0.825
        
        #self.allegro_hand= AllegroHand()

    def get(self, obs, height): 
        # Get representations 
        # episode_repr, expert_reprs = self.get_representations(obs)

        # # # print('episode_repr.shape: {}, expert_reprs.shape: {}'.format(episode_repr.shape, expert_reprs.shape))

        # all_rewards = []
        # # cost_matrices = []
        # best_reward_sum = - sys.maxsize
        # for expert_id, expert_repr in enumerate(expert_reprs):
            # # expert_repr = expert_repr.unsqueeze(0) # It should have dimension 1 for the 1st dimension
            # print('expert_repr.shape: {}, episode_Repr.shape: {}'.format(
            #     expert_repr.shape, episode_repr.shape
            # ))
            
        #     rewards= -(self.commanded_joint_angles - self.allegro_hand.get_joint_position())
        print("Height is ", height)
        
        rewards= -60*np.exp(-np.linalg.norm((height-self.fixed_height))/0.01)
        

        #rewards=height.numpy()
            
        rewards *= self.sinkhorn_rew_scale

            # all_rewards.append(rewards)
            # sum_rewards = np.sum(rewards)
            #     #cost_matrices.append(cost_matrix)
            # if sum_rewards > best_reward_sum:
            #     best_reward_sum = sum_rewards 
            #     best_expert_id = expert_id

        #final_reward = all_rewards[best_expert_id]
        #final_cost_matrix = cost_matrices[best_expert_id]
        best_reward_sum=rewards

        return best_reward_sum ,0, 0