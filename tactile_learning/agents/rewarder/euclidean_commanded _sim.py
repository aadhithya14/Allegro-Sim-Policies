#Rewarder module for sinkhorn cosine

import numpy as np
import torch 
import sys
from holobot.robot.allegro.allegro import AllegroHand
from tactile_learning.utils import cosine_distance, optimal_transport_plan

from .rewarder import Rewarder

class EuclideanCommandedSim(Rewarder):
    def __init__(
            self,
            commanded_joint_angles,
            exponential_weight_init=False, 
            **kwargs
    ):
        super().__init__(**kwargs)
        self.exponential_weight_init = exponential_weight_init
        self.commanded_joint_angles=commanded_joint_angles
        self.allegro_hand= AllegroHand()

    def get(self, obs): 
        # Get representations 
        episode_repr, expert_reprs = self.get_representations(obs)

        # print('episode_repr.shape: {}, expert_reprs.shape: {}'.format(episode_repr.shape, expert_reprs.shape))

        all_rewards = []
        cost_matrices = []
        best_reward_sum = - sys.maxsize
        for expert_id, expert_repr in enumerate(expert_reprs):
            # expert_repr = expert_repr.unsqueeze(0) # It should have dimension 1 for the 1st dimension
            print('expert_repr.shape: {}, episode_Repr.shape: {}'.format(
                expert_repr.shape, episode_repr.shape
            ))
            
            rewards= -(self.commanded_joint_angles - self.allegro_hand.get_joint_position())
           
            rewards *= self.sinkhorn_rew_scale

            all_rewards.append(rewards)
            sum_rewards = np.sum(rewards)
            #cost_matrices.append(cost_matrix)
            if sum_rewards > best_reward_sum:
                best_reward_sum = sum_rewards 
                best_expert_id = expert_id

        final_reward = all_rewards[best_expert_id]
        #final_cost_matrix = cost_matrices[best_expert_id]

        return final_reward, best_expert_id