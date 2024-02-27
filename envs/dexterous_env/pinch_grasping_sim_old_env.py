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
from holobot.constants import *
from plots.data_logging import Log, ListOfLogs, NoLog, SimpleLog 
from torch_utils import quat_mul, quat2mat, orientation_error,orientation_error_from_quat, axisangle2quat
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import time
import torch
from .env import DexterityEnv
from .camera_stream import SimCameraStreamEnv
XELA_NUM_SENSORS = 15 # 3 in thumb 4 in other 3 fingers 
XELA_NUM_TAXELS = 16 

from holobot.utils.network import ZMQCameraPublisher, ZMQCompressedImageTransmitter,ZMQKeypointPublisher,ZMQKeypointSubscriber, ZMQCameraSubscriber

#from isaacgymenvs.tasks.base.vec_task import VecTask
#from isaacgymenvs.tasks.base.vec_task import VecTask
#gym=gymapi.aquire_gym()

#@hydra.main(version_base = '1.2', config_path = 'configs', config_name = 'envs')
class PinchGraspingSim(DexterityEnv):

        def __init__(self, num_envs =1,num_per_row = 6,spacing = 2.5,show_axis=0, cam_pose=gymapi.Vec3(0,1,0), env_path=None, log_file=None,log_conf={},full_log=False,env_suite='banana', flag=0, control_mode= 'Position_Velocity', is_kinova=False,**kwargs):#gymapi.Vec3(2,4,5)):
                
                self.sim_params = gymapi.SimParams()
                self.physics_engine=gymapi.SIM_PHYSX
                self.gym=gymapi.acquire_gym()
                print(num_envs)
                self.num_envs=num_envs
                self.is_kinova=False

                super().__init__(**kwargs)
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
                self.actor_indices=[]
                self.device="cuda:0"
                self.axes_geom = gymutil.AxesGeometry(0.1)
                
                self.sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
                self.sphere_pose = gymapi.Transform(r=self.sphere_rot)
                self.sphere_geom = gymutil.WireframeSphereGeometry(0.03, 12, 12, self.sphere_pose, color=(1, 0, 0))
                self.env_suite=env_suite
                self.gymtorch=gymtorch
                self.control_mode=control_mode
                self.cnt=2
                self._stream_oculus= True

                # self.rgb_publisher = ZMQCameraPublisher(
                # host = '172.24.71.211',
                # port = 10005
                # )
                
                # if self._stream_oculus:
                #         self.rgb_viz_publisher = ZMQCompressedImageTransmitter(
                #                 host = '172.24.71.211',
                #                 port = 10005 + VIZ_PORT_OFFSET
                #         )

                # self.depth_publisher = ZMQCameraPublisher(
                # host = '172.24.71.211',
                # port = 10005 + DEPTH_PORT_OFFSET 
                # )

        # set common parameters
                self.sim_params.dt = self.dt= 1/60
                self.sim_params.substeps = 2
                self.sim_params.up_axis = gymapi.UP_AXIS_Z
                self.sim_params.gravity = gymapi.Vec3(0.0, -9.8, 0)
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
                self.sim = self.gym.create_sim(self.compute_device_id,1, self.physics_engine, self.sim_params)
                #self.sim_device="cuda:0"

        # Add ground
                self.plane_params = gymapi.PlaneParams()
                self.gym.add_ground(self.sim, self.plane_params)
                
        # create viewer #
                #self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
                self.viewer= None
                #Get the camera pose and place the camera there
                self.cam_pose=cam_pose
                if self.viewer is None:
                        print("*** Failed to create viewer")
                        #quit()
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
                self.asset_root = "/home/aadhithya/AllegroSim/urdf/"
                self.asset_file = "allegro_hand_description/urdf/model_only_hand.urdf"
                self.table_asset= "allegro_hand_description/urdf/table.urdf"
                self.cube_asset= "allegro_hand_description/urdf/cube_multicolor.urdf"
                self.sponge_asset= "allegro_hand_description/urdf/eraser.urdf"
                
                print("Loading asset '%s' from '%s'" % (self.asset_file, self.asset_root)) 
                # Loads the asset
                self.asset = self.gym.load_urdf(self.sim, self.asset_root, self.asset_file, self.asset_options)
                self.table_asset = self.gym.load_urdf(self.sim, self.asset_root, self.table_asset, self.table_asset_options)
        
                self.object_asset_options = gymapi.AssetOptions()
                self.object_asset= self.gym.load_urdf(self.sim, self.asset_root, self.cube_asset,self.object_asset_options)
               
                
                self.num_dofs=self.get_dof_count()
                print("Num DOFS", self.num_dofs)
                #self.action_space= Box(-1.0, 1.0, (self.num_envs,self.num_dofs))
                #self.observation_space= Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8)
                self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device='cpu')
                self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device='cpu')
                self.actuated_dof_indices = [i for i in range(self.num_dofs)]
                self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
                
                self.env_path="/home/aadhithya/AllegroSim/"  
                # Call Function to create and load the environment              
                self.load()
                self.actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
                self.root_state_tensor = gymtorch.wrap_tensor(self.actor_root_state_tensor)
                print("Root state tensor",self.root_state_tensor)
                self.root_state_tensor = self.root_state_tensor.view(-1, 13)
                self.object_indices=to_torch(self.object_indices, dtype=torch.int32,device='cpu')
                #print(self.object_indices)
                # for i in range(self.num_envs):
                       
                #         self.root_state_tensor[self.object_indices[i],0:3]=to_torch([1,1.3,0.06],dtype=torch.float,device='cpu')
                #         self.root_state_tensor[self.object_indices[i],3:7]=to_torch([-1.3,-0.707, 0,0],dtype=torch.float,device='cpu')
                       
                       
                # #print(self.root_state_tensor)
                # self.gym.set_actor_root_state_tensor_indexed(self.sim,
                #                                        gymtorch.unwrap_tensor(self.root_state_tensor),
                #                                        gymtorch.unwrap_tensor(self.object_indices), len(self.object_indices))
                #self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
                self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
                self.rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
                #self.init_log(self.env_path,self.log_file,self.full_log,self.log_conf)
                print("Initialisation complete")
                #self.reset()
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
                self.object_handles=[]
                sensor_pose = gymapi.Transform()

                print("Loading Assets")
                
                self.mids=0.5 * (self.upper_limits + self.lower_limits)
                for i in range(self.num_envs):
        
                        self.env = self.gym.create_env(self.sim, self.env_lower, self.env_upper, self.num_per_row)
                        print("env Created")
                        self.envs.append(self.env)
                        
                        transform = gymapi.Transform()
                        transform.p = gymapi.Vec3(0,1,0)
                        transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(0))
                        self.camera_props = gymapi.CameraProperties()
                        self.camera_props.horizontal_fov = 100
                        self.camera_props.width = 480
                        self.camera_props.height = 480
                        self.camera_props.enable_tensors = True
                        self.camera_handle = self.gym.create_camera_sensor(self.env, self.camera_props)
                        camera_position = gymapi.Vec3(0.8,1, 0.01) 
                        camera_target = gymapi.Vec3(1,0.9, 0.01)
                        self.gym.set_camera_location(self.camera_handle, self.envs[i], camera_position, camera_target)
                        self.camera_handles.append(self.camera_handle)
                        self.gym.start_access_image_tensors(self.sim)       
                                
                
                                
                        if self.is_kinova is True:
                                self.actor_pose = gymapi.Transform()
                                self.actor_pose.p = gymapi.Vec3(1,0.96, 0.0)
                                self.actor_pose.r = gymapi.Quat(-0.707,0.707, 0,0)
                        else:
                               
                                self.actor_pose = gymapi.Transform()
                                self.actor_pose.p = gymapi.Vec3(1,0.93, 0.0)
                                self.actor_pose.r = gymapi.Quat(-0.707,0.707, 0,0)

                        self.table_pose = gymapi.Transform()
                        self.table_pose.p = gymapi.Vec3(0.7, 0.0, 0.3)
                        self.table_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
                        self.actor_handle = self.gym.create_actor(self.env, self.asset, self.actor_pose, "actor", i, 1)
                        self.table_handle = self.gym.create_actor(self.env, self.table_asset, self.table_pose, "table", i, 1)
                        
                       
                        self.object_pose = gymapi.Transform()
                        self.object_pose.p = gymapi.Vec3()
                        pose_dy, pose_dz = -0.05, -0.05
                        pose_dx=0

                        self.object_pose.p.x= self.actor_pose.p.x + pose_dx
                        self.object_pose.p.y = self.actor_pose.p.y + pose_dy
                        self.object_pose.p.z = self.actor_pose.p.z + pose_dz
                        #self.object_handle = self.gym.create_actor(self.envs[i], self.object_asset,self.object_pose, "sponge",i, 0, 0)
                                                        # self.actor_handle = self.gym.create_actor(self.env, self.asset,self.actor_pose, "actor", i, 1)
                        self.object_handle = self.gym.create_actor(self.envs[i], self.object_asset,self.object_pose, "cube",i, 0, 0)
                        self.object_handles.append(self.object_handle)

                       
                        object_idx = self.gym.get_actor_index(self.env, self.object_handle, gymapi.DOMAIN_SIM)
                        self.object_indices.append(object_idx)                        
                        actor_idx = self.gym.get_actor_index(self.env, self.actor_handle, gymapi.DOMAIN_SIM)
                        self.actor_indices.append(actor_idx)

                        self.attractor_handles[i] = []
                        #self.dof_handle=self.gym.find_actor_dof_handle(self.env, self.actor_handle, 'kinova_link_5')
                        
                        body_dict = self.gym.get_actor_rigid_body_dict(self.env, self.actor_handle)
                        self.actor_handles.append(self.actor_handle)
                            
                        #self.gym.set_actor_dof_properties(self.envs[i], self.actor_handles[i], self.dof_props)

                        self.dof_states = self.gym.get_actor_dof_states(self.envs[i], self.actor_handles[i], gymapi.STATE_NONE)
                        if self.is_kinova is True:
                                for j in range(self.num_dofs+13):   
                                        if j!=23 and j!=28 and j!=18 and j!=13 : 
                                                self.gym.set_rigid_body_color(self.env, self.actor_handles[i],j, gymapi.MESH_VISUAL, gymapi.Vec3(0.15, 0.15, 0.15)) 
                        else:        
                                for j in range(self.num_dofs+13):   
                                        if j!=20 and j!=15 and j!=10 and j!=5 : 
                                                self.gym.set_rigid_body_color(self.env, self.actor_handles[i],j, gymapi.MESH_VISUAL, gymapi.Vec3(0.15, 0.15, 0.15))
                        # else:
                       
                        self.props = self.gym.get_actor_dof_properties(self.envs[i], self.actor_handles[i])
                        self.home_position=torch.zeros((self.num_envs,self.num_dofs),dtype=torch.float32, device='cpu')
               
                        self.props["stiffness"] =[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]#0.5,0.0001,0.0001,0.0001,0.00001,0.0001, 0.0001,0.0001,0.0001 ,0.0001,0.0001,0.0001]#[3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01,3.00000268e+01]  #[1500,1500,1500,1000,1000,1000,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]#[1500,1500,1500,1000,1000,1000,3,3,0.0001,0.0001,0.01,0.0001,0.0001,0.0001,0.00001,0.0001, 0.0001,0.0001,0.0001 ,0.0001,0.0001,0.0001]#[1500,1500,1500,1000,1000,1000,3,3,0.0001,0.0001,0.01,0.0001,0.0001,0.0001,0.00001,0.0001, 0.0001,0.0001,0.0001 ,0.0001,0.0001,0.0001]#[1500,1500,1500,1000,1000,1000,2,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001, 0.0001,0.0001,0.0001 ,0.0001,0.0001,0.0001]#[1500,1500,1500,1000,1000,1000,2,2,2,0.00,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001, 0.0001,0.0001,0.0001 ,0.0001,0.0001,0.0001]
                #props["stiffness"] = [1.5,2.0,1.8,1.3,1.5,2.0,1.8,1.3, 1.5 , 2.0, 18, 1.3 , 2.2 ,1.7 ,1.9 ,1.3]
                        self.props["damping"] = [0.1,0.1,0.1,0.1,0.1,0,1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]#[ 1.57186347e-05, 1.57186347e-05, 1.57186347e-05 , 1.57186347e-05 ,1.57186347e-05 , 1.57186347e-05, 1.57186347e-05, 1.57186347e-05 ,1.57186347e-05 ,1.57186347e-05, 1.57186347e-05, 1.57186347e-05 ,1.57186347e-05 ,1.57186347e-05,1.57186347e-05,1.57186347e-05,1.57186347e-05,1.57186347e-05,1.57186347e-05, 1.57186347e-05,1.57186347e-05, 1.57186347e-05]#[ 100, 100, 100 , 100 ,100 , 100, 0.000001, 0.000001 ,0.000001 ,0.000001, 0.000001, 0.000001 ,0.000001 ,0.000001,0.000001,0.000001,0.000001,0.000001,0.000001, 0.000001,0.000001, 0.000001]#[ 100, 100, 100 , 100 ,100 , 100, 0.000001, 0.000001 ,0.000001 ,0.000001, 0.000001, 0.000001 ,0.000001 ,0.000001,0.000001,0.000001,0.000001,0.000001,0.000001, 0.000001,0.000001, 0.000001]#[ 0, 0, 0 , 0 ,0 ,0,0.1 , 0.11 ,0.12 ,0.13, 0.1, 0.11 ,0.12 , 0.13, 0.1, 0.11 , 0.12 ,0.13,  0.17 , 0.2 ,0.12, 0.11]
                        self.props["friction"] = [0.01]*16
                        self.props["armature"] = [0.001]*16
                        
                        self.set_control_mode(self.control_mode)

                        self.gym.set_actor_dof_properties(self.envs[i], self.actor_handles[i], self.props) 
                        
                axes_geom = gymutil.AxesGeometry(0.1)

                sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
                sphere_pose = gymapi.Transform(r=sphere_rot)
                sphere_geom = gymutil.WireframeSphereGeometry(0.03, 12, 12, sphere_pose, color=(1, 0, 0))
                
                #print(props)
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
                self.obs={}
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
                self.current_angles=[]
                
                        
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
                                        
                        self.gym.step_graphics(self.sim)
                        self.gym.render_all_camera_sensors(self.sim)

                        self.gym.draw_viewer(self.viewer, self.sim, False)

                self.object_indices=to_torch(self.object_indices, dtype=torch.int32,device='cpu')
                #print(self.object_indices)
                for i in range(self.num_envs):
                        self.root_state_tensor[self.object_indices[i],0:3]=to_torch([0.94,0.85,0],dtype=torch.float,device='cpu')
                        #self.root_state_tensor[self.object_indices[i],0:3]=to_torch([0.93,0.85,0],dtype=torch.float,device='cpu')
                        self.root_state_tensor[self.object_indices[i],3:7]=to_torch([-0.707,-0.707, 0,0],dtype=torch.float,device='cpu')
                        
                #print(self.root_state_tensor)
              
                # for i in range(5):
                
                        
                #         self.gym.simulate(self.sim)
                #         self.gym.fetch_results(self.sim, True)
                #         self.gym.refresh_dof_state_tensor(self.sim)
                                        
                #         self.gym.step_graphics(self.sim)
                #         self.gym.render_all_camera_sensors(self.sim)

                #         self.gym.draw_viewer(self.viewer, self.sim, False)


                        
                        #self.gym.sync_frame_time(self.sim)
                self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                       gymtorch.unwrap_tensor(self.root_state_tensor),
                                                       gymtorch.unwrap_tensor(self.object_indices), len(self.object_indices))    
                
                for i in range(60):
                
                        
                         self.gym.simulate(self.sim)
                         self.gym.fetch_results(self.sim, True)
                         self.gym.refresh_dof_state_tensor(self.sim)
                                        
                         self.gym.step_graphics(self.sim)
                         self.gym.render_all_camera_sensors(self.sim)

                         self.gym.draw_viewer(self.viewer, self.sim, False)
                        
                #time.sleep(1)   
                self.observation = 'image'    
                self.state=self.compute_observation(observation=self.observation)
                self.features = 'position'
                self.state_pos = self.compute_observation(self.features)

               
                # print("State", self.state)
                # print("state_pos", self.state_pos)
                #print(self.state)
                #print("cube position")
                self.obs['pixels']=self.state
                self.obs['features']=self.state_pos
                # curr_sensor_values = np.zeros((XELA_NUM_SENSORS, XELA_NUM_TAXELS, 3)) 
                # tactile_readings = dict(
                #         sensor_values = curr_sensor_values,
                #         timestamp = curr_sensor_values
                # )
                # self.obs['tactile']=curr_sensor_values

                # hello world 
                #self.log.add('state',self.state.tolist())
                
                
                return self.obs
                #self.move()

        def return_log(self):
                return self.log  

        def set_home_state(self):  
                self.home_state = torch.zeros(self.num_envs,16) 
                for i in range(self.num_envs):  
                        #self.gym.set_actor_dof_properties(self.envs[i], self.actor_handles[i], self.props)              
                        #self.home_position[i]=torch.tensor([4.1252675 , 3.7878363 , 0.99253345 ,1.9849404 , 1.543693 ,  4.564504, 4.4979942e-03, -2.2780743e-01,  7.2583640e-01,  7.5492060e-01,9.9997884e-01,  4.4602406e-01,  4.6249545e-01,  8.0273646e-01,-1.5198354e-04, -2.3233567e-01,  7.3347342e-01,  7.4528450e-01,8.9934438e-02, -1.5348759e-01,  8.2852399e-01,  7.6638561e-01])
                        if self.is_kinova is True:
                                self.home_state[i]=torch.tensor([4.1252675 , 3.7878363 , 0.99253345 ,1.9849404 , 1.543693 ,  4.564504,-0.00137183, -0.22922094,  0.7265581 ,  0.79128325,0.9890924 ,  0.37431374,  0.36866143,
                                                        0.77558154,  0.00662423,
                                                        -0.23064502,  0.73253167,  0.7449019 ,  0.08261403, -0.15844858,
                                                        0.82595366,  0.7666822 ])
                        else:
                                self.home_state[i]=torch.tensor([-0.00137183, -0.22922094,  0.7265581 ,  0.79128325,0.9890924 ,  0.37431374,  0.36866143,
                                                        0.77558154,  0.00662423,
                                                        -0.23064502,  0.73253167,  0.7449019 ,  0.08261403, -0.15844858,
                                                        0.82595366,  0.7666822 ])  

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
        
        def calc_reward(self, base_action):
                
                self.nextposition = self.compute_observation('position')
                self.reward_val= np.exp(-np.linalg.norm((base_action-self.nextposition))/100)
                return self.reward_val
        
        def height_reward(self,current_height,fixed_height):
                self.reward_val= -np.exp(-np.linalg.norm((current_height-fixed_height))/0.01)
                return self.reward_val

        
        
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
                self.actions=to_torch(self.actions,dtype=torch.float, device='cpu') 
                #print(self.upper_limits)
                
                #print(self.num_envs)
                
                self.next_state=np.zeros(self.num_dofs)
                #self.gym.prepare_sim(self.sim)
                self.next_update_time = 0.1
                self.frame = 0
             
                #self.cube_pose.p = gymapi.Vec3(1.3,self.pose_y , 0.1)
                
                #while not self.gym.query_viewer_has_closed(self.viewer):
                t = self.gym.get_sim_time(self.sim)
                # self.log.add('true_state', self.actions.tolist())
                
                
                
                for e in range(self.num_envs):
                       
                        for k in range(self.num_dofs):
                              self.allegro_pos[k]=self.gym.get_dof_position(self.envs[e],k)  
                              self.kinova_vel[k]=self.gym.get_dof_velocity(self.envs[e],k)

                # print('self.actions: {}'.format(self.actions))             
                self.set_position(self.actions)

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
                # self.update_log()
                for i in range(self.num_dofs):
                        #self.state[i]=self.gym.get_dof_position(self.env,i)
                        self.next_state[i]=self.gym.get_dof_position(self.env,i)
                        self.next_velocity[i]=self.gym.get_dof_velocity(self.env,i)
               
                self.observation='image'
                self.nextstate=self.compute_observation(self.observation)
                self.next_obs='position'
                self.nextposition = self.compute_observation(self.next_obs)
                self.current_angles.append(self.nextposition)
                # print("NextState")
                # print(self.nextstate)
                self.next_state=to_torch(self.next_state,device='cpu')
                #self.rigid_body_index=self.gym.find_asset_rigid_body_index(self.asset, "allegrokinova")
                self.obs={}

                
                self.obs['pixels']=self.nextstate
                self.obs['features']=self.nextposition                

                
               
                self.reward, self.done, infos =0 , False, {'is_success': False} 
                self.gym.clear_lines(self.viewer)
                self.gym.refresh_rigid_body_state_tensor(self.sim)

                
                print("Done")

                return self.obs,self.done,self.reward, infos

        def compute_observation(self, observation):
                self.gym.refresh_dof_state_tensor(self.sim)
                self.gym.refresh_actor_root_state_tensor(self.sim)
                self.gym.refresh_rigid_body_state_tensor(self.sim) 

                if observation=='image':
                        for i in range(1):
                            self.color_image =self.gym.get_camera_image_gpu_tensor(self.sim,self.envs[i], self.camera_handles[i], gymapi.IMAGE_COLOR)
                            self.color_image=self.gymtorch.wrap_tensor(self.color_image)
                            self.color_image=self.color_image.cpu().numpy()
                            self.color_image=self.color_image[:,:,[0,1,2]]
                            
                        state= np.transpose(self.color_image, (2,0,1))
                    
           
               

                elif observation=='position':
                        
                        state=np.zeros(self.num_dofs)
                        for i in range(self.num_dofs):
                                state[i]=self.gym.get_dof_position(self.env,i)  

                elif observation=='velocity':
                        state=np.zeros(self.num_dofs)
                        for i in range(self.num_dofs):
                                state[i]=self.gym.get_dof_velocity(self.env,i) 
                       
                elif observation=='full_state':
                        for i in range(2*self.num_dofs):
                                if i<self.num_dofs:
                                        state[i]=self.gym.get_dof_position(self.env,i)  
                                else:
                                        state[i]=self.gym.get_dof_velocity(self.env,i)  


                #cv2.imwrite(,self.image)
                #self.gym.destroy_viewer(self.viewer)
                #self.gym.destroy_sim(self.sim)
                return state



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
        
        def get_current_angles(self):
                return self.current_angles

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

        def render(self, mode='rbg_array', width=0, height=0):
                return self.get_rgb_images()

        def get_rgb_images(self):
            self.color_image = None
            while self.color_image is None:
                # Obtaining and aligning the frame
                #print(frame)  
                #self.env.gym.render_all_camera_sensors(self.env.sim)
                for i in range(1):
                            self.color_image =self.gym.get_camera_image_gpu_tensor(self.sim,self.envs[i], self.camera_handles[i], gymapi.IMAGE_COLOR)
                            self.color_image=self.gymtorch.wrap_tensor(self.color_image)
                            self.color_image=self.color_image.cpu().numpy()
                            self.color_image=self.color_image[:,:,[2,1,0]]
                            
                            self.depth_image =self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i],self.camera_handles[i],gymapi.IMAGE_DEPTH)
                            self.depth_image =self.gymtorch.wrap_tensor(self.depth_image)
                            self.depth_image=self.depth_image.cpu().numpy()
                            timestamp=self.get_time()
                    
            return self.color_image
            

        def _get_curr_image(self, visualize=True):
                image, _ = self.get_rgb_images()
                image = cv2.cvtColor(image)
                image = image.fromarray(image, 'RGB')
                if visualize:
                        img = self.visualize_image_transform(image)
                        img = np.asarray(img)
                else:
                        img = self.image_transform(image)
                        img = torch.FloatTensor(img)
                return img # NOTE: This is for environment     
        
        def get_object_height(self):
                for i in range(self.num_envs):
                        current_height= self.root_state_tensor[self.object_indices[i],1]
                return current_height
        # def _init_camera_subscribers(self):
        #         self._rgb_streams, self._depth_streams = [], []

        #         for idx in range(1):
                       
        #                         self._rgb_streams.append(ZMQCameraSubscriber(
        #                         host = '172.24.71.211',
        #                         port = 10005 + idx,
        #                         topic_type = 'RGB'
        #                         ))

        #                         self._depth_streams.append(ZMQCameraSubscriber(
        #                         host = '172.24.71.211',
        #                         port =10005 + idx + DEPTH_PORT_OFFSET,
        #                         topic_type = 'Depth'
        #                        ))

        # def get_image(self):
        #         images = [stream.recv_rgb_image() for stream in self._rgb_streams]
        #         return images

        # def get_depth_images(self):
        #         images = [stream.recv_depth_image() for stream in self._depth_streams]
        #         return images
        

        

                        

       
        