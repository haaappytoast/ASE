from enum import Enum
import numpy as np
import torch

from isaacgym import gymapi
from isaacgym import gymtorch

from env.tasks.humanoid import Humanoid
from utils import gym_util
from utils.motion_lib import MotionLib, DeepMimicMotionLib
from isaacgym.torch_utils import *

from utils import torch_utils
import os

class testHumanoid(Humanoid):
    class StateInit(Enum):      
        Default = 0
        Start = 1
        Random = 2

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        
        ##
        self._motion_dt = 2 * sim_params.dt
        ##
        
        #! random state initialization set
        state_init = cfg["env"]["stateInit"]                                
        self._state_init = testHumanoid.StateInit[state_init]                

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)                                 

        #! set reference motion
        self._motion_ids = torch.zeros(self.num_envs, device=self.device)
        self._motion_times = torch.zeros(self.num_envs, device=self.device)
        self._phase = torch.zeros((self.num_envs, 1), device=self.device)
        motion_file = cfg['env']['motion_file']                             
        self._load_motion(motion_file)                                      


        # for debug
        # self.get_asset_joint_dict()
        # self.get_asset_joint_type()
        return

    # debug
    def get_humanoid_asset(self, env_ids=None):
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        print("env_ids size: ", env_ids)
        
        for env_id in env_ids:
            env_ptr = self.envs[env_id]

            asset_root = self.cfg["env"]["asset"]["assetRoot"]
            asset_file = self.cfg["env"]["asset"]["assetFileName"]

            asset_path = os.path.join(asset_root, asset_file)
            asset_root = os.path.dirname(asset_path)
            asset_file = os.path.basename(asset_path)

            asset_options = gymapi.AssetOptions()
            asset_options.angular_damping = 0.01
            asset_options.max_angular_velocity = 100.0
            #! Default mode used to actuate Asset joints ->  lets the joints move freely within their range of motion
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE 
            humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        return humanoid_asset

    # debug
    # this function should be used after initializing humanoid first
    def get_asset_joint_dict(self, env_ids=None):
        humanoid_asset = self.get_humanoid_asset()
        jnt_dict = self.gym.get_asset_joint_dict(humanoid_asset)
        return jnt_dict
    
    # debug
    def get_asset_joint_type(self, env_ids=None):
        humanoid_asset = self.get_humanoid_asset()
        
        for jnt_idx, jnt_name in enumerate(self.get_asset_joint_dict()):
            jnt_type = self.gym.get_asset_joint_type(humanoid_asset, jnt_idx)
            print("joint name: %s, joint_type: %s" %(jnt_name, jnt_type))
        return
    
    def _setup_character_props(self, key_bodies):
        
        #! TODO: deepmimic use axis-angle rotation representation, but use 6D first
        asset_file = self.cfg["env"]["asset"]["assetFileName"]

        if (asset_file == "mjcf/amp_humanoid.xml"):
            #! root_h + num_body * (pos, rot, vel, ang_vel) - root_pos
            self._dof_body_ids = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]                      
            self._dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
            self._num_actions = 28                                                              

            self._num_obs = 1 + 15 * (3 + 6 + 3 + 3) - 3 + 1

        return
    
    def pre_physics_step(self, actions):
        #! for test
        ####
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        motion_len = self._motion_lib._motion_lengths
        motion_ids = self._motion_ids
        motion_times = torch.remainder(self.progress_buf * self._motion_dt, motion_len)
        print("motion_times:", motion_times.item())

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
            = self._motion_lib.get_motion_state(motion_ids, motion_times)

        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)


        env_ids_int32 = self._humanoid_actor_ids[env_ids]

        # three of them all the same
        # print("\n\n===========================\n\n")
        # print("post_physics_step after/ self._humanoid_root_states: ", self._humanoid_root_states)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                    gymtorch.unwrap_tensor(self._root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self._dof_state),
                                                gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # return super().pre_physics_step(actions)
        return


    def post_physics_step(self):
        self.progress_buf+=1
        time_elapsed = self._motion_times + self.progress_buf * self.dt
        self._phase =  self._motion_lib._calc_phase(self._motion_ids, time_elapsed.to(self.device)).view(self.num_envs, 1)
        self._refresh_sim_tensors()
        self._compute_observations()
        # self._compute_reward(self.actions)
        self._compute_reset()

        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        _rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs

        rigid_body_state_reshaped = _rigid_body_state.view(self.num_envs, bodies_per_env, 13)  #! <- (1, 272, 13)

        # _rigid_body_pos/rot/vel/ang_vel share same data_prt with self._rigid_body_state & self.rigid_body_state_reshaped
        self._rigid_body_pos = rigid_body_state_reshaped[..., :self.num_bodies, 0:3]            # torch.Size([1, 15, 3])    # 15: humanoid, 17: with sword
        self._rigid_body_rot = rigid_body_state_reshaped[..., :self.num_bodies, 3:7]            # torch.Size([1, 15, 3])
        self._rigid_body_vel = rigid_body_state_reshaped[..., :self.num_bodies, 7:10]           # torch.Size([1, 15, 3])
        self._rigid_body_ang_vel = rigid_body_state_reshaped[..., :self.num_bodies, 10:13]      # torch.Size([1, 15, 3])


        body_pos2 = self._rigid_body_pos
        body_rot2 = self._rigid_body_rot
        body_vel2 = self._rigid_body_vel
        body_ang_vel2 = self._rigid_body_ang_vel
        
        print("body_pos2: ", body_pos2.shape, body_pos2)
        print("body_rot2: ", body_rot2.shape, body_rot2)
        print("body_vel2: ", body_vel2.shape, body_vel2)
        print("body_ang_vel2: ", body_ang_vel2.shape, body_ang_vel2)
        ####

        self.extras["terminate"] = self._terminate_buf

        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        return

    def _load_motion(self, motion_file):
        #! have to be changed
        assert(self._dof_offsets[-1] == self.num_dof)
        self._motion_lib = DeepMimicMotionLib(motion_file=motion_file,
                                     dof_body_ids=self._dof_body_ids,
                                     dof_offsets=self._dof_offsets,
                                     key_body_ids=self._key_body_ids.cpu().numpy(), 
                                     device=self.device)
        return

    def _reset_actors(self, env_ids):
        if (self._state_init == testHumanoid.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == testHumanoid.StateInit.Start
              or self._state_init == testHumanoid.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        return
    
    def _reset_default(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        
        self._motion_ids = 0
        self._motion_times = 0
        self._phase[env_ids] = 0
        return

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs) 
        
        if (self._state_init == testHumanoid.StateInit.Random):
            motion_times = self._motion_lib.sample_time(motion_ids)
        elif (self._state_init == testHumanoid.StateInit.Start):
            motion_times = torch.zeros(num_envs, device=self.device)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
            
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)

        dof_vel = self._motion_lib._get_body_local_angvel(motion_ids, motion_times)
        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)
        
        self._motion_ids = motion_ids
        self._motion_times = motion_times
        self._phase[env_ids] = self._motion_lib._calc_phase(motion_ids, motion_times).view(num_envs, 1)
        return
    
    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        # since, self._root_states share same memory -> self._root_state도 같이 바뀐다.    
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel
        
        # since _dof_pos share same memory with self._dof_state -> self._dof_state도 같이 바뀜
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel
        return
    
    def _compute_humanoid_obs(self, env_ids=None):
        if (env_ids is None):
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
            _phase = self._phase
        else:
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            body_vel = self._rigid_body_vel[env_ids]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]
            _phase = self._phase[env_ids]
        
        obs = build_deepmimic_observations(body_pos, body_rot, body_vel, body_ang_vel, self._local_root_obs,
                                                self._root_height_obs, _phase)
        return obs

    def _get_humanoid_ref_reward_obs(self):
        body_orn, body_ang_vel = self._motion_lib.get_motion_body_state(self._motion_ids, self._motion_times)
        return body_orn, body_ang_vel
    
    def _compute_reward(self, actions):
        global_rot_ref, global_ang_vel_ref = self._get_humanoid_ref_reward_obs()
        obs_ref= build_deepmimic_reward_observation(global_rot_ref, global_ang_vel_ref).to(self.device)
        obs = build_deepmimic_reward_observation(self._rigid_body_rot, self._rigid_body_ang_vel)
        orn_offset = 4*self.num_bodies
        ang_vel = self._rigid_body_ang_vel
        orn_reward = self._compute_orn_reward(obs[:, :orn_offset], obs_ref[:, :orn_offset])
        ang_vel_reward = self._compute_ang_vel_reward(obs[:,orn_offset:], obs_ref[:,orn_offset:])
        return 0
    
    def _compute_orn_reward(self, obs, obs_ref):
        
        return 0
    
    def _compute_ang_vel_reward(self, obs_ang_vel, obs_ref_ang_vel):
        
        reward = calculate_humanoid_ang_vel_reward(obs_ang_vel, obs_ref_ang_vel)
        return reward
#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def build_deepmimic_observations(body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs, phase_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool, Tensor) -> Tensor
    
    root_pos = body_pos[:, 0, :]    
    root_rot = body_rot[:, 0, :]    
    root_h = root_pos[:, 2:3]       
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)   
    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))   
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                                    heading_rot_expand.shape[2])        
    
    root_pos_expand = root_pos.unsqueeze(-2)           
    local_body_pos = body_pos - root_pos_expand         
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])    
    flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)        
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2]) 
    local_body_pos = local_body_pos[..., 3:] # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])  
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot) 
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot) 
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])
    
    if (local_root_obs):
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])  
    flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)                          
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])  
    
    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])   
    flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)                                       
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])   
    
    # num_obs = 1 + (3 * 14) + (6 * 15) + (3 * 15) + (3 * 15) + 1  = 224
    obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel, phase_obs), dim=-1)
    # obs = torch.cat((base_obs, phase_obs), dim=0)
    
    return obs

@torch.jit.script
def build_deepmimic_reward_observation(body_rot, body_ang_vel):
    # type: (Tensor, Tensor) -> Tensor
    root_rot = body_rot[:, 0, :] 
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_rot.shape[1], 1))   
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                                    heading_rot_expand.shape[2])     
    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])  
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot).reshape(body_rot.shape[0], -1)
    
    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])   
    flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel).reshape(body_ang_vel.shape[0], -1)                                       

    obs = torch.cat((flat_local_body_rot, flat_local_body_ang_vel), dim=-1)
    return obs

@torch.jit.script
def calculate_humanoid_orn_reward(obs_buf, obs_ref_buf):
    # type: (Tensor, Tensor) -> Tensor
    reward = torch.ones_like(obs_buf[:, 0]) 
    return reward

@torch.jit.script
def calculate_humanoid_ang_vel_reward(obs_buf, obs_ref_buf):
    # type: (Tensor, Tensor) -> Tensor
    reward = torch.exp(-0.05*torch.sum((obs_buf- obs_ref_buf)**2, dim=1))
    return reward