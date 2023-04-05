# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from enum import Enum
import numpy as np
import torch
from torch import Tensor

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil

from env.tasks.humanoid import Humanoid, dof_to_obs, compute_grot_from_lrot, dof_to_local_rotation
from utils import gym_util
from utils.motion_lib import DeepMimicMotionLib
from isaacgym.torch_utils import *

from utils import torch_utils
import sys
from poselib.poselib.core import *

class HumanoidTest(Humanoid):
    class StateInit(Enum):
        Start = 1
        Random = 2

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        state_init = cfg["env"]["stateInit"]
        #! from ase/data/cfg
        self._state_init = HumanoidTest.StateInit[state_init]

        self._usePhase = True

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        #! set reference motion
        self._env_ids = torch.range(0, self.num_envs-1, device=self.device, dtype=torch.int64)
        self._motion_ids = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
        self._motion_times = torch.zeros(self.num_envs, device=self.device)

        self.num_ref_obs = (12 * 4) + 28 + 3 + (4 * 3) + 3
        self.ref_buf = torch.zeros((self.num_envs, self.num_ref_obs), device=self.device, dtype=torch.float)

        self._dof_buf = torch.zeros((self.num_envs, 48 + 28), device=self.device, dtype=torch.float)

        self._key_body_pos = torch.zeros((self.num_envs, len(self._key_body_ids), 3), device=self.device, dtype=torch.float)
        self._com_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)

        motion_file = cfg['env']['motion_file']
        self._load_motion(motion_file)

        self.is_train = self.cfg["args"].train
        return  

    def _create_envs(self, num_envs, spacing, num_per_row):
        super()._create_envs(num_envs, spacing, num_per_row)

        # get mass from xml file
        rbody_prop = self.gym.get_actor_rigid_body_properties(self.envs[0], self.humanoid_handles[0])
        self.body_mass = []
        for i in range(len(rbody_prop)):
            self.body_mass.append(rbody_prop[i].mass)
        self.body_mass = torch.tensor(self.body_mass, dtype=torch.float, device=self.device)
        return

    def post_physics_step(self):

        self.progress_buf += 1
        self.ones = torch.ones(self._motion_times.shape).to(self.device)
        self._motion_times += self.ones * self.dt

        self._refresh_sim_tensors()
        self._compute_observations()

        #! compute reference observation                
        self._compute_ref_observations()

        self.visualize_com()
        
        self.actions = None
        self._compute_reward(self.actions)
        self._compute_reset()
        
        self.extras["terminate"] = self._terminate_buf

        return

    def visualize_com(self):
        # debug viz
        if self.viewer and self.is_train is not True:
            self._update_debug_viz()
            # draw height lines
            # self.gym.refresh_rigid_body_state_tensor(self.sim)
            sphere_geom = gymutil.WireframeSphereGeometry(0.1, 16, 16, None, color=(1, 0, 0))
            
            for i in range(self.num_envs):
                base_pos = (self._com_pos[i, :]).cpu().numpy()
                x = base_pos[0]
                y = base_pos[1]
                z = base_pos[2]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)         
        return

    def _load_motion(self, motion_file):
        assert(self._dof_offsets[-1] == self.num_dof)
        self._motion_lib = DeepMimicMotionLib(motion_file=motion_file,
                                     dof_body_ids=self._dof_body_ids,
                                     dof_offsets=self._dof_offsets,
                                     key_body_ids=self._key_body_ids.cpu().numpy(), 
                                     device=self.device)
        return
    
    # humanoid.py의 self.reset에서 실행이 되는데 이 때, env_ids를 tensor로 바꿔주는 코드가 들어있음
    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        
        # done_indices가 있는 것! -> humanoid state, terminate, progress buffer, motion_times 등등을 reset해줌
        super()._reset_envs(env_ids)
        #! compute reference observation
        if (len(env_ids)> 0):
            self._init_ref_obs(env_ids)
        return
    
    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        
    def _init_ref_obs(self, env_ids):
        # 이것도 새로운 motion_times에 맞게 reference observation 바꿔줘야됌
        # 이미 motion_times는 _reset_envs()안에서 reset되어 calculate 되어짐
        self._compute_ref_observations(env_ids) # env_ids에 대하여 observation 구해주고
        return

    
    def _setup_character_props(self, key_bodies):
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)    # data/cfg/train

        if (asset_file == "mjcf/amp_humanoid.xml"):
            self._dof_body_ids = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14] #! body that has joints attached!
            self._dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
            self._dof_obs_size = 72     #! 6 (joint_obs_size) * 12 (num_joints)
            self._num_actions = 28      #! num_dof
                            #! root_h + num_body * (pos, rot, vel, ang_vel) - root_pos
            self._num_obs = (3 * 15) + (4 * 15) + (3 * 15) + (3 * 15)
            self._parent_indices = [-1,  0,  1,  1,  3,  4,  1,  6,  7,  0,  9, 10, 0, 12, 13]

        else:
            print("Unsupported character config file: {s}".format(asset_file))
            assert(False)

        return
    
    def _compute_ref_observations(self, env_ids=None):
        ref_obs = self._compute_ref_obs(env_ids)
        if (env_ids is None):
            self.ref_buf[:] = ref_obs
        else:
            # print("3. reset된 env에 대해서 buffer에 ref_observation 값 넣어주기 : ", env_ids, "\n\n\n")
            self.ref_buf[env_ids] = ref_obs
        
        return
    
    #! state 다시 initialize 해주는 코드!
    def _reset_actors(self, env_ids):
        if (self._state_init == HumanoidTest.StateInit.Start
              or self._state_init == HumanoidTest.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        return

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)
                
        if (self._state_init == HumanoidTest.StateInit.Random):
            # motion_times = self._motion_lib.sample_time(motion_ids, self.cfg["env"]["episodeLength"], self.dt, train_epoch, self.is_train)
            motion_times = self._motion_lib.sample_time(motion_ids)
        elif (self._state_init == HumanoidTest.StateInit.Start):
            motion_times = torch.zeros(num_envs, device=self.device)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        
        # root_pos.shape:  torch.Size([1, 3], [1, 4], [1, 28], [1, 3], [1, 4, 3] )
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
                = self._motion_lib.get_motion_state(motion_ids, motion_times) 

        # reset humanoid state
        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        # for reference motion        
        self._reset_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times

        if (env_ids is None):        # 환경 1개 일때
            self._motion_times[:] = motion_times
        else:                        # 환경 여러 개일 때
            self._motion_times[env_ids] = motion_times
        return

    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        if(self.is_train):
            self._humanoid_root_states[env_ids, 0:3] = root_pos
        else:
            self._humanoid_root_states[env_ids, 0:3] = root_pos + torch.tensor([0, 0, 0.05]).to(device=self.device)        
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel
        
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel
        return

    def _compute_observations(self, env_ids=None):
        obs = self._compute_humanoid_obs(env_ids)
        if (env_ids is None):
            self._dof_buf[:] = obs[:, :76]
            self._com_pos[:] = obs[:, 76:79]
            self.obs_buf[:] = obs[:, 79:]

            self._key_body_pos = self._rigid_body_pos[:, self._key_body_ids]
        else:
            self._dof_buf[env_ids] = obs[:, :76]
            self._com_pos[env_ids] = obs[:, 76:79]
            self.obs_buf[env_ids] = obs[:, 79:]

            self._key_body_pos[env_ids] = self._rigid_body_pos[env_ids.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]
        return
        
    def _compute_humanoid_obs(self, env_ids=None):
        if (env_ids is None):
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
            dof_pos = self._dof_pos
            dof_vel = self._dof_vel
        else:
            body_pos = self._rigid_body_pos[env_ids]            # [num_envs, 15, 3]
            body_rot = self._rigid_body_rot[env_ids]            # [num_envs, 15, 4]
            body_vel = self._rigid_body_vel[env_ids]            # [num_envs, 15, 3]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]    # [num_envs, 15, 3]
            dof_pos = self._dof_pos[env_ids]                    # [num_envs, num_dof]
            dof_vel = self._dof_vel[env_ids]                    # [num_envs, num_dof]

        # obs = compute_humanoid_dof_observation(body_pos, body_rot, body_vel, body_ang_vel, self._local_root_obs,
        #                                         self._root_height_obs, dof_pos, dof_vel)
        obs = compute_humanoid_observations(body_pos, body_rot, body_vel, body_ang_vel, dof_pos, dof_vel, self.body_mass)
        
        return obs

    # 여기서 motion_times랑 motion_ids reset된 걸로 해야되는 건가? -> 확인해보기
    def _compute_ref_obs(self, env_ids=None):
        # post_physics_step에서 compute_ref_observation() 불렀을 때
        if (env_ids is None):
            local_dof_pos, local_dof_vel, global_ee_pos, global_root \
                = self._motion_lib.get_motion_state_for_reference(self._motion_ids, self._motion_times)
            body_pos = self._motion_lib._get_body_pos(self._motion_ids, self._motion_times)
        else:
            local_dof_pos, local_dof_vel, global_ee_pos, global_root \
                = self._motion_lib.get_motion_state_for_reference(self._reset_ref_motion_ids, self._reset_ref_motion_times)
            body_pos = self._motion_lib._get_body_pos(self._reset_ref_motion_ids, self._reset_ref_motion_times)

        # local_lrot = dof_to_local_rotation(local_dof_pos, (len(self._dof_offsets) - 1) * 4, dof_offsets=self._dof_offsets)
        local_dof_pos = local_dof_pos[:, self._dof_body_ids]
        flat_local_lrot = local_dof_pos.reshape(local_dof_pos.shape[0], len(self._dof_body_ids) * local_dof_pos.shape[2]) 
        flat_global_ee_pos = global_ee_pos.reshape(global_ee_pos.shape[0], global_ee_pos.shape[1] * global_ee_pos.shape[2])                     # [num_envs, 4  * 3]
        flat_global_root = global_root.reshape(global_root.shape[0], global_root.shape[1] * global_root.shape[2])

        com_pos = self._compute_com(body_pos, self.body_mass)

        # [num_envs, 94] = (12 * 4)           + 28                 + (4 * 3)     + 3       + 3
        ref_obs = torch.cat((flat_local_lrot, local_dof_vel, flat_global_ee_pos, com_pos, flat_global_root), dim=-1)
        return ref_obs

    def _compute_reward(self, actions):
        obs = self._dof_buf              # shape: [num_envs, 196]
        ref_obs = self.ref_buf           # shape: [num_envs, 117]
        key_pos = self._key_body_pos
        self.rew_buf[:] = compute_deepmm_reward(obs, ref_obs, key_pos, self._com_pos, len(self._dof_offsets)-1)
        return

    def _compute_com(self, body_states, body_masses):
        """Compute center-of-mass position"""
        com_pos = compute_com(body_states, body_masses)
        return com_pos

#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def compute_com(body_pos, body_masses):
    # type: (Tensor, Tensor) -> Tensor
    num_bodies = body_pos.shape[1]
    total_mass = torch.sum(body_masses)  #

    mass_expand = body_masses.unsqueeze(-1) # [num_bodies] -> [num_bodies, 1]
    mass_expand = mass_expand.repeat((body_pos.shape[0], 1))                                        # [num_envs * num_bodies]
    flat_mass = mass_expand.reshape(body_pos.shape[0], body_pos.shape[1])                           # [num_envs, num_bodies]

    flat_body_pos = body_pos.reshape(body_pos.shape[0] * body_pos.shape[1], body_pos.shape[2])
    flat_com = torch.mul(mass_expand, flat_body_pos)                                                # [num_envs * num_bodies, 3]
    com = flat_com.reshape(body_pos.shape[0], body_pos.shape[1], body_pos.shape[2])                 # [num_envs, num_bodies, 3]
    com_pos = torch.sum(com, dim=-2) / total_mass                                                   # [num_envs, 3]

    return com_pos

@torch.jit.script
def compute_humanoid_dof_observation(body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs, dof_pos, dof_vel):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool, Tensor, Tensor) -> Tensor

    root_pos = body_pos[:, 0, :]    # torch.Size([1, 3])
    root_rot = body_rot[:, 0, :]    # torch.Size([1, 4])
    root_h = root_pos[:, 2:3]       # get z-value
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)   # quat from heading to ref_dir(global x-axis)
    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h                     # [num_envs, 1]
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))   # shape: [1, 15, 4]
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                                    heading_rot_expand.shape[2])        # shrink shape: [1, 15, 4] -> [15, 4]
    
    root_pos_expand = root_pos.unsqueeze(-2)            # shape: [1, 1, 3]
    local_body_pos = body_pos - root_pos_expand         #! root_relative_position / shape: [1, 15, 3] / 15: num_body
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])    # shrink shape: [15, 3]/ 15: num_body
    flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)        #! root의 local x-axis에서 바라본 root_relative_position of link / shape: [1, 15, 3]
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])    # [1, 15 * 3]

    _dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
    dof_lrot = dof_to_local_rotation(dof_pos, 48, _dof_offsets)    # [num_envs, 4 * 12]
    
    cuda = torch.device('cuda')
    dof_lrot.to(cuda)
    obs = torch.cat((dof_lrot, dof_vel, local_body_pos, root_h_obs), dim=-1)
    # obs = torch.cat(([body_lrot]), dim=-1)
    return obs

@torch.jit.script
def compute_humanoid_observations(body_pos, body_rot, body_vel, body_ang_vel, dof_pos, dof_vel, body_masses):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor

    root_pos = body_pos[:, 0, :]    # torch.Size([1, 3])
    root_rot = body_rot[:, 0, :]    # torch.Size([1, 4])
    root_h = root_pos[:, 2:3]       # get z-value
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)   # quat from heading to ref_dir(global x-axis)
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))   # shape: [1, 15, 4]
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                                    heading_rot_expand.shape[2])        # shrink shape: [1, 15, 4] -> [15, 4]
    
    root_pos_expand = root_pos.unsqueeze(-2)            # shape: [1, 1, 3]
    local_body_pos = body_pos - root_pos_expand         #! root_relative_position / shape: [1, 15, 3] / 15: num_body
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])    # shrink shape: [15, 3]/ 15: num_body
    flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)        #! root의 local x-axis에서 바라본 root_relative_position of link / shape: [1, 15, 3]
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])    # [1, 15 * 3]

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])  # shape: [15, 4]
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot) #! local(root coordinate)에서 바라본 body rot / shape: [num_envs * 15,4]
    local_body_rot = flat_local_body_rot.reshape(body_rot.shape[0], body_rot.shape[1] * body_rot.shape[2])

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])  # torch.Size([15, 3])
    flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)                          #! local(root coordinate)에서 바라본 velocity torch.Size([15, 3])
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])  # torch.Size([1, 15 * 3])
    
    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])   # torch.Size([15, 3])
    flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)                                       #! local(root coordinate)에서 바라본 velocity torch.Size([15, 3])
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])   # torch.Size([1, 15 * 3])
    
    _dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
    dof_lrot = dof_to_local_rotation(dof_pos, 48, _dof_offsets)    # [num_envs, 4 * 12]
    
    cuda = torch.device('cuda')
    dof_lrot.to(cuda)
    
    com_pos = compute_com(body_pos, body_masses) # [num_envs, 3]
    #                   (48)    + (28)  + (3)       + (3 * 15)      + (4 * 15)      + (3 * 15)      + (3 * 15)
    obs = torch.cat((dof_lrot, dof_vel, com_pos, local_body_pos, local_body_rot, local_body_vel, local_body_ang_vel), dim=-1)
    # obs = torch.cat(([body_lrot]), dim=-1)
    return obs


@torch.jit.script
def compute_deepmm_reward(obs_buf, ref_buf, sim_key_pos, com_pos, num_joints):
    # type: (Tensor, Tensor, Tensor, Tensor, int) -> Tensor
    num_envs = obs_buf.shape[0]
    num_key_body = 4
    pose_w = 0.65
    vel_w = 0.15
    ee_w = 0.1
    com_w = 0.1
    #### 1. local_dof rotation
    # get simulated character's local_body_rot_obs
    local_dof_pos = obs_buf[:, 0:48]                                                   # [num_envs, 12 * 4]
    local_dof = local_dof_pos.reshape(num_envs * num_joints, -1)                       # [num_envs * num_joints, 4]

    # get reference character's local_dof_pos
    ref_local_dof_pos = ref_buf[:, 0:48]                                               # [num_envs, 12 * 4]
    ref_local_dof_pos = ref_local_dof_pos.reshape(num_envs * num_joints, -1)           # [num_envs, num_joints, 4]
    
    # get quaternion difference
    inv_local_dof = quat_inverse(local_dof)
    dof_diff = quat_mul_norm(ref_local_dof_pos, inv_local_dof)    

    # get scalar rotation of a quaternion about its axis in radians 
    rot_diff_angle, rot_diff_axis = quat_angle_axis(dof_diff)                          # [num_envs * 12], [num_envs * 12, 3]
    flat_rot_diff_angle = rot_diff_angle.reshape(num_envs, -1)

    sum_rot_diff_angle = torch.sum(flat_rot_diff_angle**2, dim=-1)
    
    pose_reward = torch.exp(-0.20 * sum_rot_diff_angle)
    
    #### 2. local_dof velocity
    
    # get simulated character's dof_vel
    local_dof_vel = obs_buf[:, 48:48 + 28]                                             # [num_envs, 28]
        
    # get reference character's dof_vel
    ref_local_dof_vel = ref_buf[:, 48:48 + 28]                                         # [num_envs, 28]

    # get angular velocity difference
    diff_dof_vel = torch.abs(local_dof_vel - ref_local_dof_vel)                        # [num_envs, 28]      
    sum_diff_dof_vel = torch.sum(diff_dof_vel**2, dim=-1)                              # [num_envs]
    
    vel_reward = torch.exp(-0.008 * sum_diff_dof_vel)

    #### 3. global end effector position
    # get simulated character's ee position
    global_ee_key_pos = sim_key_pos
    flat_global_ee_key_pos = global_ee_key_pos.reshape(num_envs, -1)                   # [num_envs, 12]

    # get reference character's ee position
    ref_global_ee_key_pos = ref_buf[:, 76:88]
    diff_ee_pos = flat_global_ee_key_pos - ref_global_ee_key_pos
    sum_diff_ee_pos = torch.sum(diff_ee_pos**2, dim=-1)

    ee_reward = torch.exp(-8 * sum_diff_ee_pos)
    
    #### 4. get com difference
    # get simulated character's com_pos
    sim_com_pos = com_pos

    # get reference character's com_pos
    ref_com_pos = ref_buf[:, 88:91]

    # com_pos difference
    diff_com_pos = torch.abs(sim_com_pos - ref_com_pos)
    sum_diff_com_pos = torch.sum(diff_com_pos, dim=-1)
    com_reward = torch.exp(-10 * sum_diff_com_pos)

    # reference charater's global root position
    reward = pose_w * pose_reward + vel_w * vel_reward + ee_w * ee_reward + com_w * com_reward
    print("pose_reward: ", pose_reward)
    print("vel_reward: ", vel_reward)
    print("ee_reward: ", ee_reward)
    print("com_reward: ", com_reward)
    print("---------------------------------------------\n")
    return reward
