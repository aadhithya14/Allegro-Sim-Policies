from abc import ABC, abstractmethod

import numpy as np
#from utils import clamp, AssetDesc
import math
import hydra
from copy import copy
from franka_allegro.utils.inverse_kinematics import qpos_from_site_pose
from franka_allegro.utils.min_jerk import generate_joint_space_min_jerk
#import gym
#from gym.spaces import Box
# from franka_allegro.franka_allegro import FrankaAllegro
from franka_allegro.franka import Dishwasher
import dmc2gymnasium
from openteach.components import Component
from openteach.utils.timer import FrequencyTimer
from openteach.utils.network import ZMQCameraPublisher, ZMQCompressedImageTransmitter,ZMQKeypointPublisher,ZMQKeypointSubscriber
from openteach.components.environment.hand_env import Hand_Env
from openteach.constants import *
from scipy.spatial.transform import Rotation, Slerp
import PIL
import time
import zmq

import cv2 
import os
import mujoco

class Mujoco_Franka_Hand(DexterityEnv):
    def __init__(self):
        
        # Create the environment
        self.env= dmc2gymnasium.DMCGym('franka_allegro_dishwasher','move')

    def reset(self):
        self.env.reset()
        self.previous_endeff=np.array([ 0.06020084,  0.37755416  ,0.25784193,  0.08539103  ,0.64307946 ,-0.33063455,0.68544728])
        self.last_hand_joint = np.array([-0.122244 ,-0.0362273 ,-0.0116873 ,-0.00222757 ,-0.000240485 ,-0.0359609 ,-0.011603 ,-0.00221183 ,-0.000243734 ,-0.035978 ,-0.0116102 ,-0.00221351 ,0.261164 ,7.21134e-05 ,0.000836664 ,0.000192601])
        self.cnt=0

    
    def step(self,action):
       
        if joint_angles is None:
            joint_angles=self.last_hand_joint
        else:
            self.last_hand_joint = joint_angles
        
        if endeff_positions is not None:
            self.previous_endeff=np.around(endeff_positions,2)

        position=self.previous_endeff[0:3]
        quat=self.previous_endeff[3:7]
        joint_angles = np.nan_to_num(joint_angles, nan=0.0, posinf=0.0, neginf=0.0)
        action=self.previous_endeff
        IKResult = qpos_from_site_pose(self.env.physics,site_name="attachment_site",target_pos=position,target_quat=quat,joint_names=["joint7","joint6","joint5","joint4","joint3","joint2","joint1"], max_steps= 100,joint_limits=np.array([self.env.action_space.low,self.env.action_space.high]))
        
        IKResult_dict=IKResult._asdict()
        qpos=IKResult_dict['qpos']
        arm_pos=IKResult_dict['qpos'][0:7]
        joint_pos_from_ik=np.concatenate((arm_pos,joint_angles))
        self.env.step(joint_pos_from_ik[0:23])



    def reset(self):
        self.env.reset()

    def _rot2quat(self, rot_mat):
        # Here we will use the resolution scale to set the translation resolution
        R = Rotation.from_matrix(
            rot_mat).as_quat()
        return R
    
    def _rot2cartMuJoCo(self,mat):
        quat = np.zeros((4, 1))
        mat = mat.reshape(9,1)
        mujoco.mju_mat2Quat(quat, mat)
        return quat[:,0]

    def render(self):
        height=480
        width=480
        frame=np.hstack([self.env.physics.render(height, width, camera_id=0)])
        return frame

    def get_rgb_depth_images(self):
        color_image=self.env.physics.render(480,480,camera_id=0)
        color_image=color_image[:,:,[2,1,0]]
        # For now no depth is being returned 
        depth=0
        time_value=time.time()
        return color_image,depth,time_value

    def close(self):
        pass

    def get_endeff_position(self):
        position= self.env.physics.named.data.site_xpos["attachment_site"]
        rot=self.env.physics.named.data.site_xmat["attachment_site"].reshape(3,3)
        quat=self._rot2cartMuJoCo(rot)
        cart=np.concatenate((position,quat))
        return cart

    def seed(self):
        pass

    def get_observation(self):
        pass

    def get_reward(self):
        pass

    def get_done(self):
        pass

    def get_info(self):
        pass

    def get_action_space(self):
        pass

    def get_observation_space(self):
        pass

    def get_reward_range(self):
        pass

    def get_dof_position(self):
        position= self.env.physics.joint_pos()[7:23]
        return position
    
    @property              
    def timer(self):
        return self._timer
       