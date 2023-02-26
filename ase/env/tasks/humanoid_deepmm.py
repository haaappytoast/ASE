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

from env.tasks.humanoid import Humanoid, dof_to_obs
from utils import gym_util
from utils.motion_lib import DeepMimicMotionLib
from isaacgym.torch_utils import *

from utils import torch_utils
import sys
from poselib.poselib.core import *

class HumanoidDeepmm(Humanoid):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        state_init = cfg["env"]["stateInit"]
        #! from ase/data/cfg
        self._state_init = HumanoidDeepmm.StateInit[state_init]

        self._usePhase = True
        
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        #! set reference motion
        self._env_ids = torch.zeros(self.num_envs, device=self.device)
        self._motion_ids = torch.zeros(self.num_envs, device=self.device)
        self._motion_times = torch.zeros(self.num_envs, device=self.device)
        self._phase = torch.zeros(self.num_envs, device=self.device)

        self.num_ref_obs = 117
        self.ref_buf = torch.zeros((self.num_envs, self.num_ref_obs), device=self.device, dtype=torch.float)

        motion_file = cfg['env']['motion_file']
        self._load_motion(motion_file)

        return

    def post_physics_step(self):
        self.progress_buf += 1
        time_elapsed = self._motion_times + self.progress_buf * self.dt
        self._phase =  self._motion_lib._calc_phase(self._motion_ids, time_elapsed.to(self.device)).view(self.num_envs, -1)

        self._refresh_sim_tensors()
        self._compute_observations(env_ids=None)

        #! compute reference observation
        self._compute_ref_observations(env_ids=None)

        self._compute_reward(self.actions)
        self._compute_reset()
        
        self.extras["terminate"] = self._terminate_buf

        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        return


    def _load_motion(self, motion_file):
        assert(self._dof_offsets[-1] == self.num_dof)
        self._motion_lib = DeepMimicMotionLib(motion_file=motion_file,
                                     dof_body_ids=self._dof_body_ids,
                                     dof_offsets=self._dof_offsets,
                                     key_body_ids=self._key_body_ids.cpu().numpy(), 
                                     device=self.device)
        return
    
    def _reset_envs(self, env_ids):
        if (len(env_ids) > 0):
            #! humanoid_deepmm에 Initialization Strategy에 따라 ref state + humanoid state 다시 initialize 해주는 코드!
            self._reset_actors(env_ids) #! go to humanoid_*._reset_actors()
            #! reset_env also
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()
            #! compute humanoid state -> 이걸로 그냥 실행
            self._compute_observations(env_ids=None)

            #! compute reference observation
            self._compute_ref_observations(env_ids=None)
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
            self._num_obs = 1 + 15 * (3 + 4 + 3 + 3)
            
        elif (asset_file == "mjcf/amp_humanoid_sword_shield.xml"):
            self._dof_body_ids = [1, 2, 3, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16]
            self._dof_offsets = [0, 3, 6, 9, 10, 13, 16, 17, 20, 21, 24, 27, 28, 31]
            self._dof_obs_size = 78
            self._num_actions = 31
            self._num_obs = 1 + 17 * (3 + 4 + 3 + 3) - 3

        else:
            print("Unsupported character config file: {s}".format(asset_file))
            assert(False)

        return
    def _compute_ref_observations(self, env_ids=None):
        ref_obs = self._compute_ref_obs(env_ids)

        if (env_ids is None):
            self.ref_buf[:] = ref_obs
        else:
            self.ref_buf[env_ids] = ref_obs
        return
    
    #! state 다시 initialize 해주는 코드!
    def _reset_actors(self, env_ids):
        #!!  should be always changed (HumanoidAmp -> HumanoidDeepmm)
        if (self._state_init == HumanoidDeepmm.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == HumanoidDeepmm.StateInit.Start
              or self._state_init == HumanoidDeepmm.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        return
    
    def _reset_default(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids
        return

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)
        
        if (self._state_init == HumanoidDeepmm.StateInit.Random):
            motion_times = self._motion_lib.sample_time(motion_ids)
        elif (self._state_init == HumanoidDeepmm.StateInit.Start):
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
        
        self._env_ids = env_ids
        self._motion_ids = motion_ids
        self._motion_times = motion_times

        return

    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel
        
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel
        
        return
    
    def _compute_humanoid_obs(self, env_ids=None):
        if (env_ids is None):
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
        else:
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            body_vel = self._rigid_body_vel[env_ids]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]
        
        obs = compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, self._local_root_obs,
                                                self._root_height_obs)
        return obs

    def _compute_ref_obs(self, env_ids=None):
        if (env_ids is None):
            local_body_rot, local_body_angvel, global_ee_pos \
                = self._motion_lib.get_motion_state_for_reference(self._motion_ids, self._motion_times)
        
        #! check with env_ids
        else:
            local_body_rot, local_body_angvel, global_ee_pos \
                = self._motion_lib.get_motion_state_for_reference(self._motion_ids, self._motion_times)
        flat_local_body_rot = local_body_rot.reshape(local_body_rot.shape[0], local_body_rot.shape[1] * local_body_rot.shape[2])                # [num_envs, 15 * 4]
        flat_local_body_angvel = local_body_angvel.reshape(local_body_angvel.size(0), local_body_angvel.size(1) * local_body_angvel.size(2))  #! 확인 필요  # [num_envs, 15 * 3]
        flat_global_ee_pos = global_ee_pos.reshape(global_ee_pos.shape[0], global_ee_pos.shape[1] * global_ee_pos.shape[2])                     # [num_envs, 4  * 3]
        # [num_envs, 117] = 15 * 4 + 15 * 3 + 4 * 3
        ref_obs = torch.cat((flat_local_body_rot, flat_local_body_angvel, flat_global_ee_pos), dim=-1)
        return ref_obs

    def _compute_reward(self, actions):
        #! start here!
        obs = self.obs_buf              # shape: [num_envs, 233]
        ref_obs = self.ref_buf          # shape: [num_envs, 117]
        self.rew_buf[:] = compute_deepmm_reward(self.obs_buf, self.ref_buf)
        return

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    root_pos = body_pos[:, 0, :]    # torch.Size([1, 3])
    root_rot = body_rot[:, 0, :]    # torch.Size([1, 4])
    root_h = root_pos[:, 2:3]       # get z-value
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)   # quat from heading to ref_dir(global x-axis)
    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))   # shape: [1, 15, 4]
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                                    heading_rot_expand.shape[2])        # shrink shape: [1, 15, 4] -> [15, 4]
    
    root_pos_expand = root_pos.unsqueeze(-2)            # shape: [1, 1, 3]
    local_body_pos = body_pos - root_pos_expand         #! root_relative_position / shape: [1, 15, 3] / 15: num_body
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])    # shrink shape: [15, 3]/ 15: num_body
    flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)        #! root의 local x-axis에서 바라본 root_relative_position of link / shape: [1, 15, 3]
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])    # [1, 15 * 3]
    # local_body_pos = local_body_pos[..., 3:] # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])  # shape: [15, 4]
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot) #! local(root coordinate)에서 바라본 body rot / shape: [15 * num_envs, 4]
    local_body_rot_obs = flat_local_body_rot.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot.shape[1])   #shape: [1, 15 * 4]
    
    #? 어 그럼 false면 이 안에 들어가는 값은 뭐지?
    if (local_root_obs):
        local_body_rot_obs[..., 0:4] = root_rot

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])  # torch.Size([15, 3])
    flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)                          #! local(root coordinate)에서 바라본 velocity torch.Size([15, 3])
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])  # torch.Size([1, 15 * 3])
    
    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])   # torch.Size([15, 3])
    flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)                                       #! local(root coordinate)에서 바라본 velocity torch.Size([15, 3])
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])   # torch.Size([1, 15 * 3])
    
    #!! should add phase variable to observation

    # shape: [1, 196] = 1 + (3 * 15) + (4 * 15) + (3 * 15) + (3 * 15)
    obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel), dim=-1)
    return obs


@torch.jit.script
def compute_deepmm_reward(obs_buf, ref_buf):
    # type: (Tensor, Tensor) -> Tensor

    # pose reward
    num_envs = obs_buf.shape[0]
    num_rigid_body = 15
    num_key_body = 4
    pose_w = 1

    # get reference character's local_body_rot_obs
    local_body_rot_obs = obs_buf[:, 46:106]          # [num_envs, 15 * 4]
    local_body_angvel_obs = obs_buf[:, 151:196]      # [num_envs, 15 * 4]
    # global_ee_pos = obs_buf[:, 46:106]

    local_body_rot = local_body_rot_obs.reshape(num_envs * num_rigid_body, -1)           # [num_envs * rigid_body, 4]
    local_body_angvel = local_body_angvel_obs.reshape(num_envs * num_rigid_body, -1)     # [num_envs * rigid_body, 3]
    #! check it again

    # get simulated character's local_body_rot_obs
    ref_local_body_rot_obs = ref_buf[:, 0:60]          # [num_envs, 15 * 4]
    ref_local_body_angvel_obs = ref_buf[:, 60:105]     # [num_envs, 15 * 3]
    ref_global_ee_pos_obs = ref_buf[:, 105:117]        # [num_envs, 4  * 3]

    ref_local_body_rot = ref_local_body_rot_obs.reshape(num_envs * num_rigid_body, -1)           # [num_envs, rigid_body, 4]
    ref_local_body_angvel = ref_local_body_angvel_obs.reshape(num_envs * num_rigid_body, -1)     # [num_envs, rigid_body, 3]
    ref_global_ee_pos = ref_global_ee_pos_obs.reshape(num_envs * num_key_body, -1)               # [num_envs, rigid_body, 3]
    
    # get quaternion difference
    inv_local_body_rot = quat_inverse(local_body_rot)
    body_rot_diff = quat_mul_norm(inv_local_body_rot, ref_local_body_rot)

    # get scalar rotation of a quaternion about its axis in radians
    rot_diff_angle, rot_diff_axis = quat_angle_axis(body_rot_diff)

    pose_reward = torch.exp(-2 * (torch.sum(rot_diff_angle**2, dim=-1)))

    reward = pose_w * pose_reward

    return reward
