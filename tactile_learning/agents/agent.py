# Base class for the agent module
# It will set the expert demos and the encoders
import numpy as np 
import torch

from abc import ABC, abstractmethod

from tactile_learning.models import * 
from tactile_learning.utils import * 
#from tactile_learning.datasets.data_sampler.data_sampler import data_sampler
from tactile_learning.datasets import *

class Agent(ABC):
    def __init__(
        self,
        data_path,
        expert_demo_nums,
        image_out_dir, image_model_type,
        tactile_out_dir, tactile_model_type,
        **kwargs
    ):
        
        # Set each given variable to a class variable
        self.__dict__.update(kwargs)

        # Demo based parameters
        self._set_data(data_path, expert_demo_nums)

        # Get the expert demos and set the encoders
        self._set_image_transform()
        self._set_encoders(
            image_out_dir = image_out_dir, 
            image_model_type = image_model_type,
            tactile_out_dir = tactile_out_dir, 
            tactile_model_type = tactile_model_type
        )
        self._set_expert_demos()
        

    def _set_data(self, data_path, expert_demo_nums):
        self.data_path = data_path 
        self.expert_demo_nums = expert_demo_nums
        self.roots = sorted(glob.glob(f'{data_path}/demonstration_*'))
        self.data = load_data(self.roots, demos_to_use=expert_demo_nums)
        # print("Data Dict", self.data)

        
    def _set_encoders(self, image_out_dir, image_model_type, tactile_out_dir, tactile_model_type): 
        _, self.image_encoder, self.image_transform  = init_encoder_info(self.device, image_out_dir, 'image', view_num=self.view_num, model_type=image_model_type)
        # Freeze the encoders
        self.image_encoder.eval()
        for param in self.image_encoder.parameters():
            param.requires_grad = False 

    def train(self, training=True):
        self.training = training

    def _set_image_transform(self):
        self.image_act_transform = T.Compose([
            RandomShiftsAug(pad=4),
            T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS)
        ])

        self.inv_image_transform = get_inverse_image_norm() # This is only to be able to 

    def _set_expert_demos(self): # Will stack the end frames back to back
        # We'll stack the tactile repr and the image observations
        print('IN _SET_EXPERT_DEMOS')


        self.expert_demos = []
        image_obs = [] 
        #tactile_reprs = []
        actions = []
        old_demo_id = -1
        pbar = tqdm(total=len(self.data['image']['indices']))
        print("Num ",len(self.data['image']['indices']))
        print(self.data['image']['indices'])
        for step_id in range(len(self.data['image']['indices'])): 
            # Set observations
            print(step_id)
            print(len(self.data['image']['indices']))
            demo_id , image_id = self.data['image']['indices'][step_id]
            if (demo_id != old_demo_id and step_id > 0) or (step_id == len(self.data['image']['indices'])-1):
                print("IF statement getting called")
                self.expert_demos.append(dict(
                    image_obs = torch.stack(image_obs[:], 0),
                    actions = np.stack(actions[:], 0)
                ))
                image_obs = []
                actions = []
            _, image_id = self.data['image']['indices'][step_id]
            image = load_dataset_image(
                data_path = self.data_path, 
                demo_id = demo_id, 
                image_id = image_id,
                view_num = self.view_num,
                transform = self.image_transform
            )
            
            print(image.shape)
            print(image.numpy().shape)
            cv2.imwrite("/home/aadhithya/tactile-learning/images/"+str(step_id)+".png",np.transpose(image.numpy()))
            # Set actions
            _, allegro_action_id = self.data['allegro_joint_states']['indices'][step_id]
            _, allegro_commanded_action_id = self.data['allegro_actions']['indices'][step_id]
           
            allegro_action = self.data['allegro_joint_states']['values'][demo_id][allegro_action_id]
            #kinova_action = self.data['endeffector_velocities']['values'][demo_id][allegro_action_id]
            allegro_commanded_actions =self.data['allegro_actions']['values'][demo_id][allegro_commanded_action_id]
            demo_action = allegro_commanded_actions
            image_obs.append(image)
            actions.append(demo_action)
            old_demo_id = demo_id
            pbar.update(1)
            pbar.set_description('Setting the expert demos ')

        pbar.close()

    def _get_policy_reprs_from_obs(self, representation_types, image_obs=None, tactile_repr=None, features=None, ):
         # Get the representations
        reprs = []
        if 'image' in representation_types:
            print('image_obs in get_policy: {}'.format(image_obs.shape))
            print(image_obs)
            image_obs = self.image_act_transform(image_obs.float()).to(self.device)
            image_reprs = self.image_encoder(image_obs)
            print('image_reprs.shape: {}'.format(image_reprs.shape))
            reprs.append(image_reprs)

        if 'tactile' in representation_types:
            tactile_reprs = tactile_repr.to(self.device) # This will give all the representations of one batch
            reprs.append(tactile_reprs)

        if 'features' in representation_types:
            repeated_features = features.repeat(1, self.features_repeat)
            reprs.append(repeated_features.to(self.device))

        return torch.concat(reprs, axis=-1) # Concatenate the representations to get the final representations

    @abstractmethod
    def update(self, **kwargs):
        pass 

    @abstractmethod
    def act(self, **kwargs): 
        pass

    def save_snapshot(self, keys_to_save=['actor']):
        payload = {k: self.__dict__[k] for k in keys_to_save}
        return payload
    
    def repr_dim(self, type='policy'):
        representations = self.policy_representations if type=='policy' else self.goal_representations
        repr_dim = 0
        if 'tactile' in representations:
            repr_dim += self.tactile_repr.size
        if 'image' in representations:
            repr_dim += 512
        if 'features' in representations:
            repr_dim += 16 * self.features_repeat # 23 * self.features_repeat

        return repr_dim

    def load_snapshot(self, payload):
        for k, v in payload.items():
            self.__dict__[k] = v
