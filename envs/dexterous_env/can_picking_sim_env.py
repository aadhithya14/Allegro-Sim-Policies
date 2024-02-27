from isaacgym import gymapi, gymutil
import numpy as np
#from utils import clamp, AssetDesc
import math
import hydra
from copy import copy
import gym
from gym.spaces import Box
#import torch





import cv2
from plots.data_logging import Log, ListOfLogs, NoLog, SimpleLog 
from torch_utils import quat_mul, quat2mat, orientation_error,orientation_error_from_quat, axisangle2quat
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import time
import torch


#@hydra.main(version_base = '1.2', config_path = 'configs', config_name = 'envs')
class CubeRotationSim(gym.Env):
        def __init__(self, num_envs =36,num_per_row = 6,spacing = 2.5,show_axis=0, cam_pose=gymapi.Vec3(0,1,0), env_path=None, log_file=None,log_conf={},full_log=False,env_suite='banana', flag=0, control_mode= 'Position_Velocity', is_kinova=False):#gymapi.Vec3(2,4,5)):
                
                self.sim_params = gymapi.SimParams()
                self.physics_engine=gymapi.SIM_PHYSX
                self.gym=gymapi.acquire_gym()
                self.num_envs=num_envs
                self.num_per_row=num_per_row
                self.spacing=spacing
                self.show_axis=show_axis
                self.name="Allegro_Sim"
                #Log specific parameters 
                self.log_file=log_file
                self.log_conf=log_conf
                self.full_log= full_log
                #Env specific parameters
                self.env_lower = gymapi.Vec3(-self.spacing, 0.0, -self.spacing)
                self.env_upper = gymapi.Vec3(self.spacing, self.spacing, self.spacing)
                self.envs=[]
                self.actor_handles=[]
                self.attractor_handles = {}
                self.base_poses = []
                self.object_indices=[]
                self.act_moving_average=1.0
        
                self.device="cuda:0"
                self.axes_geom = gymutil.AxesGeometry(0.1)
                self.is_kinova=False

                self.sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
                self.sphere_pose = gymapi.Transform(r=self.sphere_rot)
                self.sphere_geom = gymutil.WireframeSphereGeometry(0.03, 12, 12, self.sphere_pose, color=(1, 0, 0))
                self.env_suite=env_suite
                self.gymtorch=gymtorch
                self.control_mode=control_mode
                self.cnt=2
        #Asset Descriptor
                #self.asset_descriptors = [AssetDesc("home/aadhithya/AllegroSim/urdf/allegro_hand_description/urdf/model_new.urdf", False)] 
                #,AssetDesc("urdf/allegro/urdf/allegro_hand.urdf", False)]

        # set common parameters
                self.sim_params.dt = self.dt= 1/60
                self.sim_params.substeps = 2
                self.sim_params.up_axis = gymapi.UP_AXIS_Z
                self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        # set PhysX-specific parameters
                if self.physics_engine==gymapi.SIM_PHYSX:
                        self.sim_params.physx.use_gpu = True
                        self.sim_params.physx.solver_type = 1
                        self.sim_params.physx.num_position_iterations = 6
                        self.sim_params.physx.num_velocity_iterations = 1
                        self.sim_params.physx.contact_offset = 0.01
                        self.sim_params.physx.rest_offset = 0.0
                        self.compute_device_id=1
                        self.graphics_device_id=1
                        self.asset_id=1

        # set Flex-specific parameters
                elif self.physics_engine==gymapi.SIM_FLEX:
                        self.sim_params.flex.solver_type = 5
                        self.sim_params.flex.num_outer_iterations = 4
                        self.sim_params.flex.num_inner_iterations = 20
                        self.sim_params.flex.relaxation = 0.8
                        self.sim_params.flex.warm_start = 0.5
                        self.compute_device_id=0
                        self.graphics_device_id=0
                        
        # create sim with these parameters
                print("Creating Sim")
                print(torch.cuda.is_available())
                print(torch.cuda.device_count())
                print(torch.cuda.current_device())
                self.sim = self.gym.create_sim(self.compute_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
                #self.sim_device="cuda:0"

        # Add ground
                self.plane_params = gymapi.PlaneParams()
                self.gym.add_ground(self.sim, self.plane_params)
                
        # create viewer 
                self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

                #Get the camera pose and place the camera there
                self.cam_pose=cam_pose
                if self.viewer is None:
                        print("*** Failed to create viewer")
                        quit()
        # set asset options
                self.asset_options = gymapi.AssetOptions()
                self.asset_options.fix_base_link = True
                self.asset_options.flip_visual_attachments =  False #self.asset_descriptors[self.asset_id].flip_visual_attachments
                self.asset_options.use_mesh_materials = True
                self.asset_options.disable_gravity = True
                
                self.table_asset_options = gymapi.AssetOptions()
                self.table_asset_options.fix_base_link = True
                self.table_asset_options.flip_visual_attachments = False
                self.table_asset_options.collapse_fixed_joints = True
                self.table_asset_options.disable_gravity = True

                #get asset file
                self.asset_root = "urdf/"
                self.asset_file = "allegro_hand_description/urdf/model_new.urdf"
                self.table_asset= "allegro_hand_description/urdf/table.urdf"
                self.cube_asset= "allegro_hand_description/urdf/cube_multicolor.urdf"
                self.ball_asset= "allegro_hand_description/urdf/cube_multicolor.urdf"

                self.can_asset_file = "ycb/010_potted_meat_can/010_potted_meat_can.urdf"
                self.banana_asset_file = "ycb/011_banana/011_banana.urdf"
                self.eraser_asset_file = "allegro_hand_description/urdf/eraser.urdf"
                self.mug_asset_file = "ycb/025_mug/025_mug.urdf"
                self.brick_asset_file = "ycb/061_foam_brick/061_foam_brick.urdf"

                #self.asset_file= "allegro_hand_description/urdf/model.urdf"
                print("Loading asset '%s' from '%s'" % (self.asset_file, self.asset_root)) 
                
                self.asset = self.gym.load_urdf(self.sim, self.asset_root, self.asset_file, self.asset_options)
                self.table_asset = self.gym.load_urdf(self.sim, self.asset_root, self.table_asset, self.table_asset_options)
                if self.env_suite=='cube_flipping':
                        self.object_asset_options = gymapi.AssetOptions()
                        self.object_asset= self.gym.load_urdf(self.sim, self.asset_root, self.cube_asset,self.object_asset_options)
                elif self.env_suite=='cube_rotating':
                        self.object_asset_options = gymapi.AssetOptions()
                        self.object_asset= self.gym.load_urdf(self.sim, self.asset_root, self.cube_asset,self.object_asset_options)

                elif self.env_suite=='can_picking':
                        self.object_asset_options = gymapi.AssetOptions()
                        self.object_asset= self.gym.load_urdf(self.sim, self.asset_root, self.can_asset_file,self.object_asset_options)

                elif self.env_suite=='banana':
                        self.object_asset_options = gymapi.AssetOptions()
                        self.object_asset= self.gym.load_urdf(self.sim, self.asset_root, self.banana_asset_file,self.object_asset_options)
                
                elif self.env_suite=='eraser_turning':
                        self.object_asset_options = gymapi.AssetOptions()
                        self.object_asset= self.gym.load_urdf(self.sim, self.asset_root, self.eraser_asset_file,self.object_asset_options)

                
                self.num_dofs=self.get_dof_count()
                self.action_space= Box(-1.0, 1.0, (self.num_envs,self.num_dofs))
                self.observation_space= Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8)
                self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device='cpu')
                self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device='cpu')
                self.actuated_dof_indices = [i for i in range(self.num_dofs)]
                self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
                
                self.env_path="/home/aadhithya/AllegroSim/"  
                # Call Function to create and load the environment              
                self.load()
                self.actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
                #print(self.actor_root_state_tensor)
                self.root_state_tensor = gymtorch.wrap_tensor(self.actor_root_state_tensor)
                #print(self.root_state_tensor)
                self.root_state_tensor = self.root_state_tensor.view(-1, 13)
                #self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
                self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
                self.rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
                self.init_log(self.env_path,self.log_file,self.full_log,self.log_conf)
                #self.move()

        def load(self):
                
                self.dof_props=self.get_dof_properties()
                self.lower_limits = self.dof_props['lower']
                self.upper_limits = self.dof_props['upper']

                self.hand_dof_lower_limits = []
                self.hand_dof_upper_limits = []
                self.hand_dof_default_pos = []
                self.hand_dof_default_vel = []
                self.sensors = []
                self.camera_handles = []
                self.cube_handles=[]
                self.can_handles=[]
                self.banana_handles=[]
                sensor_pose = gymapi.Transform()

                
                self.mids=0.5 * (self.upper_limits + self.lower_limits)
                for i in range(self.num_envs):
        
                        self.env = self.gym.create_env(self.sim, self.env_lower, self.env_upper, self.num_per_row)
                        self.envs.append(self.env)
                        
                        transform = gymapi.Transform()
                        transform.p = gymapi.Vec3(0,1,0)
                        transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(0))
                        self.camera_props = gymapi.CameraProperties()
                        self.camera_props.horizontal_fov = 100.0
                        self.camera_props.width = 1080
                        self.camera_props.height = 1080
                        self.camera_props.enable_tensors = True
                        self.camera_handle = self.gym.create_camera_sensor(self.env, self.camera_props)
                        camera_position = gymapi.Vec3(1.3,2 , 0.1)
                        camera_target = gymapi.Vec3(1.5,1.7 , 0.1)
                        self.gym.set_camera_location(self.camera_handle, self.envs[i], camera_position, camera_target)
                        self.camera_handles.append(self.camera_handle)
                        self.gym.start_access_image_tensors(self.sim)       
                                
                          
                        pose = gymapi.Transform()
                        pose.p = gymapi.Vec3(1,1.2, 0.0)
                        pose.r = gymapi.Quat(-0.707107,0.0, 0.0, 0.707107)
                        #pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.pi) * gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.47 * np.pi) * gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0.25 * np.pi)

                        self.table_pose = gymapi.Transform()
                        self.table_pose.p = gymapi.Vec3(1, 0.0, 0.0)
                        self.table_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
                        #self.camera_handle = self.gym.create_camera_sensor(self.env, self.camera_props)
                        self.actor_handle = self.gym.create_actor(self.env, self.asset, pose, "actor", i, 1)
                        #self.table_handle = self.gym.create_actor(self.env, self.table_asset, self.table_pose, "table", i, 1)
                        
                        if self.env_suite =='cube_flipping':
                                                        self.cube_pose = gymapi.Transform()
                                                        self.cube_pose.p = gymapi.Vec3(1.3,0.835 , 0.1)
                                                        self.cube_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
                                                        self.cube_handle = self.gym.create_actor(self.envs[i], self.cube_asset,self.cube_pose, "cube",i, 1)
                                                        self.cube_handles.append(self.cube_handle)

                        if self.env_suite =='can_picking':
                                                        self.can_pose = gymapi.Transform()
                                                        self.can_pose.p = gymapi.Vec3(1.5,0.835 , 0)
                                                        self.can_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
                                                        self.can_handle = self.gym.create_actor(self.envs[i], self.can_asset,self.can_pose, "can",i, 1)
                                                        self.can_handles.append(self.can_handle)

                        if self.env_suite =='banana':
                                                        self.banana_pose = gymapi.Transform()
                                                        self.banana_pose.p = gymapi.Vec3(1.5,0.835 , 0)
                                                        self.banana_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
                                                        self.banana_handle = self.gym.create_actor(self.envs[i], self.banana_asset,self.banana_pose, "banana",i, 1)
                                                        self.banana_handles.append(self.banana_handle)
                        
                        
                        object_idx = self.gym.get_actor_index(self.env, self.actor_handle, gymapi.DOMAIN_SIM)
                        self.object_indices.append(object_idx)

                        self.attractor_handles[i] = []
                        #self.dof_handle=self.gym.find_actor_dof_handle(self.env, self.actor_handle, 'kinova_link_5')
                        
                        body_dict = self.gym.get_actor_rigid_body_dict(self.env, self.actor_handle)
                        #props = self.gym.get_actor_rigid_body_states(self.env, self.actor_handle, gymapi.STATE_POS)
                        #props = self.gym.get_actor_dof_properties(self.env, self.actor_handle)
                        #props["driveMode"].fill(gymapi.DOF_MODE_POS)
                        #props["stiffness"].fill(1000.0)
                        #props["damping"].fill(200.0)
                        #self.gym.set_actor_dof_properties(self.env, self.actor_handle, props)

                        self.actor_handles.append(self.actor_handle)
                            
                        #self.gym.set_actor_dof_properties(self.envs[i], self.actor_handles[i], self.dof_props)

                        self.dof_states = self.gym.get_actor_dof_states(self.envs[i], self.actor_handles[i], gymapi.STATE_NONE)
                        #for j in range(self.num_dofs):
                                #self.dof_states['pos'][j] = self.mids[j] - self.mids[j] * .5
                        for j in range(self.num_dofs+13):   
                                if j!=23 and j!=28 and j!=18 and j!=13 : 
                                        self.gym.set_rigid_body_color(self.env, self.actor_handles[i],j, gymapi.MESH_VISUAL, gymapi.Vec3(0.15, 0.15, 0.15)) 
                      
                       
                       
                        self.props = self.gym.get_actor_dof_properties(self.envs[i], self.actor_handles[i])
                        self.home_position=torch.zeros((self.num_envs,self.num_dofs),dtype=torch.float32, device='cpu')
               
                        self.props["stiffness"] =[1500,1500,1500,1000,1000,1000,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]#0.5,0.0001,0.0001,0.0001,0.00001,0.0001, 0.0001,0.0001,0.0001 ,0.0001,0.0001,0.0001]#[3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01]  #[1500,1500,1500,1000,1000,1000,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]#[1500,1500,1500,1000,1000,1000,3,3,0.0001,0.0001,0.01,0.0001,0.0001,0.0001,0.00001,0.0001, 0.0001,0.0001,0.0001 ,0.0001,0.0001,0.0001]#[1500,1500,1500,1000,1000,1000,3,3,0.0001,0.0001,0.01,0.0001,0.0001,0.0001,0.00001,0.0001, 0.0001,0.0001,0.0001 ,0.0001,0.0001,0.0001]#[1500,1500,1500,1000,1000,1000,2,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001, 0.0001,0.0001,0.0001 ,0.0001,0.0001,0.0001]#[1500,1500,1500,1000,1000,1000,2,2,2,0.00,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001, 0.0001,0.0001,0.0001 ,0.0001,0.0001,0.0001]
                #props["stiffness"] = [1.5,2.0,1.8,1.3,1.5,2.0,1.8,1.3, 1.5 , 2.0, 18, 1.3 , 2.2 ,1.7 ,1.9 ,1.3]
                        self.props["damping"] = [ 100, 100, 100 , 100 ,100 , 100, 0.1, 0.1 ,0.1,0.1, 0.1,0,1 , 0.1 , 0.1 ,0.1 , 0.1 ,0.1, 0.1 , 0.1, 0.1,0.1]#[ 1.57186347e-05, 1.57186347e-05, 1.57186347e-05 , 1.57186347e-05 ,1.57186347e-05 , 1.57186347e-05, 1.57186347e-05, 1.57186347e-05 ,1.57186347e-05 ,1.57186347e-05, 1.57186347e-05, 1.57186347e-05 ,1.57186347e-05 ,1.57186347e-05,1.57186347e-05,1.57186347e-05,1.57186347e-05,1.57186347e-05,1.57186347e-05, 1.57186347e-05,1.57186347e-05, 1.57186347e-05]#[ 100, 100, 100 , 100 ,100 , 100, 0.000001, 0.000001 ,0.000001 ,0.000001, 0.000001, 0.000001 ,0.000001 ,0.000001,0.000001,0.000001,0.000001,0.000001,0.000001, 0.000001,0.000001, 0.000001]#[ 100, 100, 100 , 100 ,100 , 100, 0.000001, 0.000001 ,0.000001 ,0.000001, 0.000001, 0.000001 ,0.000001 ,0.000001,0.000001,0.000001,0.000001,0.000001,0.000001, 0.000001,0.000001, 0.000001]#[ 0, 0, 0 , 0 ,0 ,0,0.1 , 0.11 ,0.12 ,0.13, 0.1, 0.11 ,0.12 , 0.13, 0.1, 0.11 , 0.12 ,0.13,  0.17 , 0.2 ,0.12, 0.11]
                        self.props["friction"] = [0.01]*22
                        self.props["armature"] = [0.001]*22
                        
                        self.set_control_mode(self.control_mode)

                        self.gym.set_actor_dof_properties(self.envs[i], self.actor_handles[i], self.props) 
                        
                axes_geom = gymutil.AxesGeometry(0.1)

                sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
                sphere_pose = gymapi.Transform(r=sphere_rot)
                sphere_geom = gymutil.WireframeSphereGeometry(0.03, 12, 12, sphere_pose, color=(1, 0, 0))
                kinova_attractors = ["kinova_end_effector"] 
                #print(props)


                for i in range(self.num_envs):
                                for j,body in enumerate(kinova_attractors):
                                        self.attractor_properties = gymapi.AttractorProperties()
                                        self.attractor_properties.stiffness =  1e6
                                        self.attractor_properties.damping = 500
                                        self.body_handle = self.gym.find_actor_rigid_body_handle(self.envs[i], self.actor_handles[i], body)
                                       
                                        self.attractor_properties.axes = gymapi.AXIS_ALL
                                        self.attractor_properties.rigid_handle = self.body_handle
                                attractor_handle = self.gym.create_rigid_body_attractor(self.envs[i], self.attractor_properties)
                                self.attractor_handles[i].append(attractor_handle)
                self.dof_state=self.gym.acquire_dof_state_tensor(self.sim) 
                #print(gymtorch.wrap_tensor(self.dof_state)[0])
                   
                #self.state=self.gym.get_dof_position()
        
        def get_dof_names(self):
                dof_names = self.gym.get_asset_dof_names(self.asset)
                return dof_names

        def get_dof_properties(self):
                dof_props = self.gym.get_asset_dof_properties(self.asset)
                return dof_props

        def get_dof_count(self):
                num_dofs = self.gym.get_asset_dof_count(self.asset)
                return num_dofs
        
        def get_dof_states(self):
                dof_states = np.zeros(self.num_dofs, dtype=gymapi.DofState.dtype)
                return dof_states

        def get_dof_positions(self):
                #self.dof_states = self.gym.get_actor_dof_states(self.envs[i], self.actor_handles[i], gymapi.STATE_NONE)
                self.position=np.zeros(self.num_dofs)
                for i in range(self.num_dofs):
                        self.position[i]=self.gym.get_dof_position(self.envs[0],i)
                return self.position
        
        def get_dof_velocities(self):
                #self.dof_states = self.gym.get_actor_dof_states(self.envs[i], self.actor_handles[i], gymapi.STATE_NONE)
                self.velocity=np.zeros(self.num_dofs)
                for i in range(self.num_dofs):
                        self.velocity[i]=self.gym.get_dof_velocity(self.envs[0],i)
                return self.velocity

        def get_dof_types(self):
                dof_types = [self.gym.get_asset_dof_type(self.asset, i) for i in range(self.num_dofs)]
                return dof_types

       

        def create_viewer(self):
                viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
                if viewer is None:
                        print("*** Failed to create viewer")
                        quit()
                return viewer

        
        def get_num_envs(self):
                return self.num_envs 
       
        def reset(self):
                #self.load()
                self.state=np.zeros(self.num_dofs)
                #print(self.num_dofs)
                self.dof_handle=np.zeros(self.num_envs)
                self.curr=1.0
                self.action=torch.zeros(self.num_envs,self.num_dofs)
                self.cnter=0
                self.allegro_pos=np.zeros(self.num_dofs)
                self.kinova_vel=np.zeros(self.num_dofs)
                self.next_velocity=np.zeros(self.num_dofs)
                #self.dof_states=np.zeros(self.num_envs)
                #print(self.num_dofs)
                
                
                        
                for i in range(self.num_envs):  
                        #self.gym.set_actor_dof_properties(self.envs[i], self.actor_handles[i], self.props)              
                        #self.home_position[i]=torch.tensor([4.1252675 , 3.7878363 , 0.99253345 ,1.9849404 , 1.543693 ,  4.564504, 4.4979942e-03, -2.2780743e-01,  7.2583640e-01,  7.5492060e-01,9.9997884e-01,  4.4602406e-01,  4.6249545e-01,  8.0273646e-01,-1.5198354e-04, -2.3233567e-01,  7.3347342e-01,  7.4528450e-01,8.9934438e-02, -1.5348759e-01,  8.2852399e-01,  7.6638561e-01])
                        if self.is_kinova is True:
                                self.home_position[i]=torch.tensor([4.1252675 , 3.7878363 , 0.99253345 ,1.9849404 , 1.543693 ,  4.564504,-0.00137183, -0.22922094,  0.7265581 ,  0.79128325,0.9890924 ,  0.37431374,  0.36866143,
                                                        0.77558154,  0.00662423,
                                                        -0.23064502,  0.73253167,  0.7449019 ,  0.08261403, -0.15844858,
                                                        0.82595366,  0.7666822 ])
                        else:
                                self.home_position[i]=torch.tensor([-0.00137183, -0.22922094,  0.7265581 ,  0.79128325,0.9890924 ,  0.37431374,  0.36866143,
                                                        0.77558154,  0.00662423,
                                                        -0.23064502,  0.73253167,  0.7449019 ,  0.08261403, -0.15844858,
                                                        0.82595366,  0.7666822 ])
                #self.gym.set_dof_position_target_tensor(self.sim,gymtorch.unwrap_tensor(self.home_position))
                self.set_position(self.home_position)
                for i in range(60):
                
                       
                        self.gym.simulate(self.sim)
                        self.gym.fetch_results(self.sim, True)
                        self.gym.refresh_dof_state_tensor(self.sim)
                                        #    for env in envs:
                                        #        gym.draw_env_rigid_contacts(viewer, env, colors[0], 0.5, True)

                                        # step rendering
                        self.gym.step_graphics(self.sim)
                        self.gym.render_all_camera_sensors(self.sim)

                        self.gym.draw_viewer(self.viewer, self.sim, False)


                        
                        #self.gym.sync_frame_time(self.sim)
               
                        
                #time.sleep(1)   
                self.observation = 'position'    
                self.state=self.compute_observation(self.observation)

                if self.env_suite=='cube_flipping':
                        forces = torch.zeros((self.num_envs, 3, 3), device='cpu', dtype=torch.float)
                        torques = torch.zeros((self.num_envs, 3, 3), device='cpu', dtype=torch.float)
                        forces[:, 2, 2] = -100.0
                        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)
               
                #self.image=self.gym.get_camera_image(self.sim,self.env,self,)
                #self.state=self.root_state_tensor[self.object_indices,0:13]#dof_state_tensor[0]
                #print(self.dof_count)
                print("State")
                #print(self.state)
                print("cube position")
               
                
                #self.log.add('state',self.state.tolist())
                
                return self.state
                #self.move()

        def return_log(self):
                return self.log        

        def compute_reward(self):
                reward=0
                return reward

        def terminal_state(self):
                done=0
                return done

        def get_mids(self):
                return self.mids.to("cpu")

        def get_lower_limits(self):
                return self.lower_limits.to("cpu")

        def get_upper_limits(self):
                return self.upper_limits.to("cpu")
        

        
        def osc(self,goal_pose,ee_pose,j_eef,ee_vel, decouple_pos_ori=False):
                
                
                self.ee_pose=ee_pose
                ee_pos=ee_pose[0:3]
                ee_quat=ee_pose[3:7]
                j_eef=j_eef.to(self.device)
                print(ee_quat)
                print(goal_pose[3:6])
                self.goal_pos=goal_pose[0:3]
                self.goal_ori_mat = quat2mat(quat_mul(axisangle2quat(goal_pose[3:6]), ee_quat))
                mm_inv = torch.inverse(self.mm.cpu()).to(self.device)
                kp=5000
                kd=100
    # Calculate error
                pos_err = self.goal_pos - ee_pos
                ori_err = orientation_error(self.goal_ori_mat, quat2mat(ee_quat))
                err = torch.cat([pos_err, ori_err], dim=0)
                print("Pos Error", pos_err)
                print("ORientation Error", ori_err)
                print("total error",err)
                # Determine desired wrench
               
                err = (kp * err - kd * ee_vel).unsqueeze(-1)
                m_eef_inv = j_eef @ mm_inv @ torch.transpose(j_eef, -1, -2)
                print(m_eef_inv)
                m_eef = torch.inverse(m_eef_inv)
                if decouple_pos_ori:
                        m_eef_pos_inv = j_eef[:, :3, :] @ mm_inv @ torch.transpose(j_eef[:, :3, :], 1, 2)
                        m_eef_ori_inv = j_eef[:, 3:, :] @ mm_inv @ torch.transpose(j_eef[:, 3:, :], 1, 2)
                        m_eef_pos = torch.inverse(m_eef_pos_inv)
                        m_eef_ori = torch.inverse(m_eef_ori_inv)
                        wrench_pos = m_eef_pos @ err[:, :3, :]
                        wrench_ori = m_eef_ori @ err[:, 3:, :]
                        wrench = torch.cat([wrench_pos, wrench_ori], dim=1)
                else:
                        wrench = m_eef @ err
                print(wrench)
                # Compute OSC torques
                u = torch.transpose(j_eef, -1, -2) @ wrench
                print("Torque is ",u)

                # Nullspace control torques `u_null` prevents large changes in joint configuration
                # They are added into the nullspace of OSC so that the end effector orientation remains constant
                # roboticsproceedings.org/rss07/p31.pdf
                """if rest_qpos is not None:
                        j_eef_inv = m_eef @ j_eef @ mm_inv
                        u_null = kd_null * -qd + kp_null * ((rest_qpos - q + np.pi) % (2 * np.pi) - np.pi)
                        u_null = mm @ u_null.unsqueeze(-1)
                        u += (torch.eye(control_dim).unsqueeze(0).to(device) - torch.transpose(j_eef, 1, 2) @ j_eef_inv) @ u_null
                """
                return u
        
        def control_ik(self,j_eef, dpose, num_envs, num_dofs, damping=0.05):
                """Solve damped least squares, from `franka_cube_ik_osc.py` in Isaac Gym.

                Returns: Change in DOF positions, [num_envs,num_dofs], to add to current positions.
                """
                #j_eef= torch.flatten(j_eef, start_dim=1)
                j_eef_T = torch.transpose(j_eef, 0, 1).to(self.device)
                lmbda = torch.eye(6).to(j_eef_T.device) * (damping ** 2)
                u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose)
                #u=torch.clip(u,-3.14,3.14)
                return u


       

        def euler_integration(initial_pose, linear_velocity, angular_velocity, time_step):
                """
                Perform Euler integration to convert end effector velocities to end effector pose.

                Parameters:
                        initial_pose: (x, y, theta) tuple, initial end effector pose (position and orientation in radians).
                        linear_velocity: (vx, vy) tuple, linear velocity in x and y directions.
                        angular_velocity: wz, angular velocity in radians per second around the z-axis (positive for counter-clockwise).
                        time_step: Time step for integration.

                Returns:
                        new_pose: (x, y, theta) tuple, updated end effector pose after integration.
                """
                x, y, theta = initial_pose
                vx, vy = linear_velocity

                # Update the position using linear velocities
                x += vx * time_step
                y += vy * time_step

                # Update the orientation using angular velocity
                theta += angular_velocity * time_step

                # Normalize theta to the range (-pi, pi)
                theta = np.mod(theta + np.pi, 2 * np.pi) - np.pi

                return x, y, theta

                # Example usage:
                initial_pose = (0.0, 0.0, 0.0)  # Initial pose (x, y, theta in radians)
                linear_velocity = (0.1, 0.2)    # Linear velocity (vx, vy)
                angular_velocity = 0.1          # Angular velocity in radians per second
                time_step = 0.1                 # Time step for integration

                new_pose = euler_integration(initial_pose, linear_velocity, angular_velocity, time_step)
                print("New end effector pose:", new_pose)


        
        def set_position_and_velocity(self, position , velocity):
               
                #print("Velocity is",velocity)
                
                #print("Position is", position)
               
                #print(name)
                #robot = self.env.get_actor("actor")
                #end_effector = self.env.get_body("actor", "kinova_end_effector")
                _jacobian=self.gym.acquire_jacobian_tensor(self.sim,"actor")
                jacobian = gymtorch.wrap_tensor(_jacobian)
                self.gym.refresh_jacobian_tensors(self.sim)
                _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
                rb_states = gymtorch.wrap_tensor(_rb_states)

                self.cnt+=0.2
                #print(jacobian.shape)
                #print(velocity)
                arm_end_eff_joint_index = self.gym.get_asset_rigid_body_dict(self.asset)["kinova_end_effector"]
                #print(arm_end_eff_joint_index)
                #print(arm_end_eff_joint_index)
                self._j_eef = jacobian[:, arm_end_eff_joint_index-1, :,:]
                #print(self._j_eef.shape)
                self.jacobian_eef=self._j_eef[0,:,:6].to(self.device)
                #print(self.jacobian_eef)
                props=self.gym.get_actor_rigid_body_states(self.env,self.actor_handle,gymapi.STATE_POS)
               
                body_dict = self.gym.get_actor_rigid_body_dict(self.env, self.actor_handle)
                handle=self.gym.get_rigid_handle(self.env,"actor","kinova_end_effector")
                #print("Handle is ", handle)
               
                current_pos=props[handle]
                current_position=np.array(current_pos[0][0].tolist())
                current_linear_vel=np.array(current_pos[1][0].tolist())
                current_angular_vel=np.array(current_pos[1][1].tolist())
                current_rot=np.array(current_pos[0][1].tolist())
                current_pose=to_torch(np.concatenate((current_position,current_rot)),dtype=torch.float, device=self.device)
                current_vel=to_torch(np.concatenate((current_linear_vel,current_angular_vel)),dtype=torch.float, device=self.device)
                #goal=velocity[0,:].reshape(1,6)*self.dt
                goal=[self.cnt, self.cnt, self.cnt]
                goal=to_torch(goal[0],dtype=torch.float, device=self.device)
                print("Goal",goal)
                #goal_pose[0:3]+=torch.tensor([1,1.2, 0.0],device=self.device)
                goal_position=torch.tensor([1,1.2, 0.0],device=self.device)+goal[0:3]
                goal_rot=(1+0.5*torch.cat([goal[3:6],torch.tensor([0],device=self.device)]))*torch.tensor([-0.707107,0.0, 0.0, 0.707107],device=self.device)
                
                endeff_pos=to_torch(self.get_cartesian_position()[0:3],dtype=torch.float,device= self.device)
                endeff_rot=to_torch(self.get_cartesian_position()[3:7],dtype=torch.float,device= self.device)
                pos_err = goal_position - endeff_pos
                print("Goal Rotation",goal_rot)
                orn_err = orientation_error_from_quat(goal_rot, endeff_rot)
                goal_pose = torch.cat([pos_err, orn_err], -1)
                #pose.r = gymapi.Quat(-0.707107,0.0, 0.0, 0.707107)
                #dof_pos=to_torch(self.get_arm_position(),dtype=torch.float,device= self.device)
                #goal_pose=dof_pos+goal_pose
                print("Goal_pose",goal_pose)
               
                goal_pose=to_torch(goal_pose,dtype=torch.float,device=self.device)
               
                
                #position_new=np.add(np.array(current_pos[0][0].tolist()),np.array(velocity[0,:3]*self.dt))
                #new_velocity=np.array([velocity[0,0],velocity[0,1],velocity[0,2],0])
                #rotation=np.add(np.array(current_pos[0][1].tolist()),0.5*new_velocity*self.dt*np.array(current_pos[0][1].tolist()))
                #pose=np.concatenate((position_new,rotation))
                #print(pose)
           
               
                

                
                #print(orn_cur)
                
                #print(orn_cur.shape)
                _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "actor")
                self.mm = gymtorch.wrap_tensor(_massmatrix)
                self.mm=self.mm[:,:6,:6]
                print(self.mm.shape)
                self.gym.refresh_mass_matrix_tensors(self.sim)
                #torque=self.osc(goal_pose,current_pose,self.jacobian_eef,current_vel, decouple_pos_ori=False)
                change=self.control_ik(self.jacobian_eef,goal_pose,self.num_envs,self.num_dofs)
               
                print("Change",change)
                dof_pos=to_torch(self.get_arm_position(),dtype=torch.float,device= self.device)
                print(dof_pos)
                dof_new=dof_pos+change
                dof_new=torch.clip(dof_new,-3.14,3.14)
                print("Final",dof_new)
                #print("Torque",torque)
                """m_inv = torch.inverse(mm)
                m_eef = torch.inverse(self.jacobian_eef @ m_inv @ torch.transpose(self.jacobian_eef, 1, 2))
                orn_cur /= torch.norm(orn_cur, dim=-1).unsqueeze(-1)
                orn_err = self.orientation_error(orn_des, orn_cur[6])
                """
                #print(self._j_eef[0,:,:])
                #inv_jacobian=np.linalg.inv(self.jacobian_eef)
                #print(inv_jacobian.shape)
                #print(velocity[0,:].shape)
                """_massmatrix = gym.acquire_mass_matrix_tensor(self.sim, "actor")
                mm = gymtorch.wrap_tensor(_massmatrix)
                mm = mm[:, :6, :6]          # only need
                m_inv = torch.inverse(mm)
                m_eef = torch.inverse(self.jacobian_eef @ m_inv @ torch.transpose(self.jacobian_eef, 1, 2))
                orn_cur /= torch.norm(orn_cur, dim=-1).unsqueeze(-1)
                orn_err = orientation_error(orn_des, orn_cur)

                pos_err = kp * (pos_des - pos_cur)

                if not args.pos_control:
                        pos_err *= 0

                dpose = torch.cat([pos_err, orn_err], -1)

                u = torch.transpose(j_eef, 1, 2) @ m_eef @ (kp * dpose).unsqueeze(-1) - kv * mm @ dof_vel
                
                # Set tensor action
                gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(u))`
                """
                
                #velocity = velocity[0,:].view(6,1)
                #velocity = velocity[0,:].reshape(6,1)
                #print(velocity.shape)
                #print(inv_jacobian)
                #joint_vel=np.matmul(inv_jacobian,velocity)
                #print(joint_vel)
                #self.dof_pos=self.get_arm_position()
                #print(self.dof_pos)
                #print("Commanded")
                #print(joint_vel)
                #print(joint_vel)
                
                for k in range(self.num_envs):
                        #self.gym.set_rigid_linear_velocity(self.env,arm_end_eff_joint_index,gymapi.Vec3(velocity[0,0],velocity[0,1],velocity[0,2]))
                        #self.gym.set_rigid_angular_velocity(self.env,arm_end_eff_joint_index,gymapi.Vec3(velocity[0,3],velocity[0,4],velocity[0,5]))
                        step=0
                        for i in range(6):
                                while step < 100/60:
                                       
                                        self.gym.set_dof_target_position(self.envs[k],i,dof_new[i])
                                        
                                                #print(joint_vel[i])
                                                #self.gym.set_dof_target_position(self.envs[k],6+i,pose[i])
                                        #self.gym.set_dof_target_velocity(self.envs[k],i,joint_vel[i])
                                        step = step + 1
                                #time.sleep(0.01)
                                
                                #val=self.gym.set_actor_rigid_body_states(self.envs[k],self.actor_handles[k],velocity[k][i],gymapi.STATE_VEL)
                                
                #allegro_names=['joint_a','joint_b','joint_c','joint_d','joint_e','joint_f', 'joint_g','joint_h', 'joint_i','joint_j', 'joint_k', 'joint_l', 'joint_m', 'joint_n', 'joint_o', 'joint_p']
                #allegro_handles=[]
                #for name in allegro_names:
                        #allegro_handle=self.gym.find_actor_dof_handle(self.env, self.actor_handle, name)
                        #allegro_handles.append(allegro_handle)
                #print(allegro_handles)
                        for i in range(16):
                                self.gym.set_dof_target_position(self.envs[k],6+i,position[k][i])
                                #print("position control")
                        #self.gym.set_dof_position_target_tensor(self.sim,  gymtorch.unwrap_tensor(position))
                #print(val)

        def step(self,action):
                self.dof_props=self.get_dof_properties()
                self.num_dofs=self.get_dof_count()
                self.get_dof_type=self.get_dof_types()
                self.dof_states=self.get_dof_states()
                self.dof_positions=self.get_dof_positions()
                self.dof_names=self.get_dof_names()
                self.actions = action
                #self.action_scale= 
                #self.get_dof_counttargets = self.action_scale * self.actions + self.default_dof_pos
                #self.viewer=self.create_viewer()
                stiffnesses = self.dof_props['stiffness']
                dampings = self.dof_props['damping']
                armatures = self.dof_props['armature']
                has_limits = self.dof_props['hasLimits']
                self.lower_limits = self.dof_props['lower']
                self.lower_limits=to_torch(self.lower_limits)
                self.upper_limits = self.dof_props['upper']
                self.upper_limits=to_torch(self.upper_limits)  
                self.mids=0.5 * (self.upper_limits + self.lower_limits)
                #print(self.lower_limits)
              
                #print(self.upper_limits)
                
                #print(self.num_envs)
                
                self.next_state=np.zeros(self.num_dofs)
                #self.gym.prepare_sim(self.sim)
                self.next_update_time = 0.1
                self.frame = 0
             
                #self.cube_pose.p = gymapi.Vec3(1.3,self.pose_y , 0.1)
                
                #while not self.gym.query_viewer_has_closed(self.viewer):
                t = self.gym.get_sim_time(self.sim)
                self.log.add('true_state', self.actions.tolist())
                
                
                
                for e in range(self.num_envs):
                       
                        for k in range(self.num_dofs):
                              self.allegro_pos[k]=self.gym.get_dof_position(self.envs[e],k)  
                              self.kinova_vel[k]=self.gym.get_dof_velocity(self.envs[e],k)
                             
                self.set_position_and_velocity(self.actions)

                for i in range(60):
                        
                        self.gym.simulate(self.sim)
                        self.gym.fetch_results(self.sim, True)
                        self.gym.refresh_dof_state_tensor(self.sim)
                        #    for env in envs:
                        #        gym.draw_env_rigid_contacts(viewer, env, colors[0], 0.5, True)

                        # step rendering
                self.gym.step_graphics(self.sim)
                self.gym.render_all_camera_sensors(self.sim)

                self.gym.draw_viewer(self.viewer, self.sim, False)
                        #self.gym.sync_frame_time(self.sim)
                self.update_log()
                for i in range(self.num_dofs):
                        #self.state[i]=self.gym.get_dof_position(self.env,i)
                        self.next_state[i]=self.gym.get_dof_position(self.env,i)
                        self.next_velocity[i]=self.gym.get_dof_velocity(self.env,i)
               
                self.observation='position'
                self.nextstate=self.compute_observation(self.observation)
                # print("NextState")
                # print(self.nextstate)
                self.next_state=to_torch(self.next_state,device='cpu')
                #self.rigid_body_index=self.gym.find_asset_rigid_body_index(self.asset, "allegrokinova")

                
               
               
                self.done=self.terminal_state()
                self.reward=self.compute_reward()
                self.gym.clear_lines(self.viewer)
                self.gym.refresh_rigid_body_state_tensor(self.sim)

                
                print("Done")

                return self.nextstate,self.done,self.reward, {}

        def compute_observation(self, observation):
                self.gym.refresh_dof_state_tensor(self.sim)
                self.gym.refresh_actor_root_state_tensor(self.sim)
                self.gym.refresh_rigid_body_state_tensor(self.sim) 

                if observation=='image':
                        
                        self.state = self.gym.get_camera_image(self.sim,self.env, self.camera_handle, gymapi.IMAGE_COLOR)
                        for i in range(self.num_envs):  
                                self.rgb_filename="cap_image_%d.png"%(i)
                                self.gym.write_camera_image_to_file(self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_COLOR, self.rgb_filename) 

                elif observation=='position':
                        
                        self.state=np.zeros(self.num_dofs)
                        for i in range(self.num_dofs):
                                self.state[i]=self.gym.get_dof_position(self.env,i)  

                elif observation=='velocity':
                        self.state=np.zeros(self.num_dofs)
                        for i in range(self.num_dofs):
                                self.state[i]=self.gym.get_dof_velocity(self.env,i) 
                       
                elif observation=='full_state':
                        for i in range(2*self.num_dofs):
                                if i<self.num_dofs:
                                        self.state[i]=self.gym.get_dof_position(self.env,i)  
                                else:
                                        self.state[i]=self.gym.get_dof_velocity(self.env,i)  


                #cv2.imwrite(,self.image)
                #self.gym.destroy_viewer(self.viewer)
                #self.gym.destroy_sim(self.sim)
                return self.state



        def init_log(self, env_path, log_file, make_full_log, log_conf):
                self.log_conf = log_conf
                if (env_path is not None) or (log_file is not None):
                        if make_full_log:
                                separate_files = True
                                if self.log_conf is not None:
                                        if 'separate_files' in self.log_conf:
                                                separate_files = self.log_conf['separate_files']
                                        save_to_file = True
                                if self.log_conf is not None:
                                        if 'save_to_file' in self.log_conf:
                                                save_to_file = self.log_conf['save_to_file']
                                        self.log = ListOfLogs(env_path + '_episodes', separate_files=separate_files)
                        else:
                                if log_file is not None:
                                        self.log = Log(log_file)
                                else:
                                        self.log = Log(env_path + '_episodes.json')
                else:
                        self.log = SimpleLog()

        def get_dof_position(self):
                self.state=self.compute_observation(observation='position')[6:]
                #self.state= =np.concatenate((self.state[0:4]) 
                #self.state=np.concatenate((self.state[0:4],self.state[12:],self.state[4:8],self.state[8:12]))
                return self.state
        
        def get_arm_position(self):
                self.state=self.compute_observation(observation='position')[0:6]
                return self.state
        
        def get_arm_velocity(self):
                self.state=self.compute_observation(observation='velocity')[0:6]
                return self.state
        
        def get_state(self):
                self.state=self.compute_observation(observation='position')
                return self.state

        def update_log(self):
                self.log.add('state', self.get_state().tolist())

        def get_gym(self):
                return self.gym
        
        def get_sim(self):
                return self.sim
        
        def get_time(self):
                return self.gym.get_elapsed_time(self.sim)

        def get_table_cartesian(self):
                self.table_handle = self.gym.find_actor_rigid_body_handle(self.env, self.table_handle, "base_link")
                self.table_pose = self.gym.get_rigid_transform(self.env, self.table_handle)
                self.table_position = [self.table_pose.p.x, self.table_pose.p.y, self.table_pose.p.z]   
                return self.table_position
        
        def get_cartesian_position(self):
                self.end_eff_handle = self.gym.find_actor_rigid_body_handle(self.env, self.actor_handle, "kinova_end_effector")
                self.end_eff_pose = self.gym.get_rigid_transform(self.env, self.end_eff_handle)
                self.end_eff_position = np.array([self.end_eff_pose.p.x, self.end_eff_pose.p.y, self.end_eff_pose.p.z])
                self.end_eff_rotation = np.array([self.end_eff_pose.r.x, self.end_eff_pose.r.y, self.end_eff_pose.r.z, self.end_eff_pose.r.w])
                self.end_eff_pos= np.concatenate((self.end_eff_position,self.end_eff_rotation))
                return self.end_eff_pos

        def get_rgb_depth_images(self):
                frame = None
                #while frame is None:
                # Obtaining and aligning the frame
                print(frame)  
                # image 
                for i in range(self.num_envs):
                                frame = self.gym.get_camera_image(self.sim,self.envs[i], self.camera_handles[i], gymapi.IMAGE_COLOR)
                                self.depth_image = self.gym.get_camera_image(self.sim, self.envs[i],self.camera_handles[i],gymapi.IMAGE_DEPTH)
                for i in range(self.num_envs):  
                                self.rgb_filename="cap_image_%d.png"%(i)
                time=self.get_time()
                print(time)
                self.gym.write_camera_image_to_file(self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_COLOR, self.rgb_filename)
                return frame, self.depth_image, time
        
        def set_position(self, position):
                self.gym.set_dof_position_target_tensor(self.sim,  gymtorch.unwrap_tensor(position))

        def set_velocity(self,velocity):
                self.gym.set_dof_velocity_target_tensor(self.sim,  gymtorch.unwrap_tensor(velocity))

        def set_control_mode(self,mode=None):
                for k in range(self.num_dofs):
                        if mode is not None:
                                if mode=='Position':
                                        self.props["driveMode"][k] = gymapi.DOF_MODE_POS
                                elif mode=='Velocity':
                                        self.props["driveMode"][k] = gymapi.DOF_MODE_VEL
                                elif mode=='Effort':
                                        self.props["driveMode"][k] = gymapi.DOF_MODE_EFFORT
                                elif mode=='Position_Velocity':
                                    
                                        self.props["driveMode"][k] = gymapi.DOF_MODE_POS   

                        else:
                                return
                        
        

        

                        

       
        