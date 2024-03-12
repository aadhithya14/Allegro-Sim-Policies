import glob
import hydra 
from omegaconf import DictConfig

from allegro_sim.datasets.utils.preprocess_sim import *
from allegro_sim.models.utils import * 
from allegro_sim.utils.augmentations import *
from allegro_sim.utils.data import *

from openteach.samplers.allegro import AllegroSampler 

def data_sampler(data_path, view_num, demo_to_sample): # NOTE: here to input the demo_num. not the demo_id !!!
    roots = sorted(glob.glob(f'{data_path}/demonstration_*'))
    print("Sampling data...")
    ## NOTE: sampled data: [demo_id, img_id, joint_state_id] (may need change to joint_commanded_state during deployment)
    demo_img_joint = []
    for demo_id, root in enumerate(roots):
        ## This is just temporal!! for the cube_flipping task
        # if demo_id in [0, 1, 2, 3, 5, 6, 7, 9, 11, 12, 13, 14, 17, 18, 19]:
        #     continue
        # Get the demo_num:
        demo_num = int(root.split('/')[-1].split('_')[-1])
        print(demo_num)
        if not demo_num in demo_to_sample: 
            continue  

        sampler = AllegroSampler(root, [view_num], 'rgb', 0.01)
        sampler.sample_data()
        #assert len(sampler.sampled_rgb_frame_idxs) == len(sampler.sampled_robot_idxs), "Sampled correctly"
        # print("Sampled Frame idxs", sampler.sampled_rgb_frame_idxs)
        for index in range(len(sampler.sampled_allegro_states)):
             demo_img_joint.append([demo_id, sampler.sampled_rgb_frame_idxs[0][index], sampler.sampled_robot_idxs[index]])
        # print("---------------------------------------------------------num of frames sampled {}, from demo_num: {}".format(len(sampler.sampled_rgb_frame_idxs[0]),demo_num)) 
    return demo_img_joint     

# demo_img_joint = data_sampler(f'/scratch/yd2032/Desktop/openteach_data/cube_flipping', 0, [25])
# print("**Why is this done 2 times??**")
# print(demo_img_joint)
# print("finished ")
