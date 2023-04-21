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

import env.tasks.humanoid as humanoid
import env.tasks.humanoid_test as humanoid_test
import env.tasks.humanoid_test_task as humanoid_test_task
from env.tasks.humanoid_test import compute_deepmm_reward

from utils import gym_util
from utils.motion_lib import DeepMimicMotionLib
from isaacgym.torch_utils import *

from utils import torch_utils
import sys
from poselib.poselib.core import *

TAR_ACTOR_ID = 1
TAR_FACING_ACTOR_ID = 2

class HumanoidHeading(humanoid_test_task.HumanoidTestTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._tar_speed_min = cfg["env"]["tarSpeedMin"]
        self._tar_speed_max = cfg["env"]["tarSpeedMax"]
        self._heading_change_steps_min = cfg["env"]["headingChangeStepsMin"]
        self._heading_change_steps_max = cfg["env"]["headingChangeStepsMax"]
        self._enable_rand_heading = cfg["env"]["enableRandHeading"]
        
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self._heading_change_steps = torch.zeros([self.num_envs], device=self.device, dtype=torch.int64)
        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._tar_speed = torch.ones([self.num_envs], device=self.device, dtype=torch.float)
        self._tar_dir = torch.zeros([self.num_envs, 2], device=self.device, dtype=torch.float)
        self._tar_dir[..., 0] = 1.0
        
        self._tar_facing_dir = torch.zeros([self.num_envs, 2], device=self.device, dtype=torch.float)
        self._tar_facing_dir[..., 0] = 1.0

        if (not self.headless):
            self._build_marker_state_tensors()

        return  

    def _build_marker_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs    # [self.num_envs * num_actors, (pos3, rot4, lin vel3, ang vel3)]
        print("inside build_marker_state: ", self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., TAR_ACTOR_ID, :])
        exit()
        self._marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., TAR_ACTOR_ID, :]
        self._marker_pos = self._marker_states[..., :3]
        self._marker_rot = self._marker_states[..., 3:7]
        
        #! 정확히 잘 모르겠음
        self._marker_actor_ids = self._humanoid_actor_ids + TAR_ACTOR_ID    # num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32)

        self._face_marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., TAR_FACING_ACTOR_ID, :]
        self._face_marker_pos = self._face_marker_states[..., :3]
        self._face_marker_rot = self._face_marker_states[..., 3:7]
        self._face_marker_actor_ids = self._humanoid_actor_ids + TAR_FACING_ACTOR_ID
        
        return
    
    def _update_task(self):
        reset_task_mask = self.progress_buf >= self._heading_change_steps   # cool time 찼으면 ㅋㅋ
        rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
        if len(rest_env_ids) > 0:
            self._reset_task(rest_env_ids)
        return
    
    def _reset_task(self, env_ids):
        n = len(env_ids)
        if (self._enable_rand_heading):
            rand_theta = 2 * np.pi * torch.rand(n, device=self.device) - np.pi                  #   -pi ~ pi
            rand_face_theta = 2 * np.pi * torch.rand(n, device=self.device) - np.pi             #   -pi ~ pi
        else:
            rand_theta = torch.zeros(n, device=self.device)
            rand_face_theta = torch.zeros(n, device=self.device)

        tar_dir = torch.stack([torch.cos(rand_theta), torch.sin(rand_theta)], dim=-1)
        tar_speed = (self._tar_speed_max - self._tar_speed_min) * torch.rand(n, device=self.device) + self._tar_speed_min
        #! 이게 왜 있는 건지는 잘 모르겠음
        change_steps = torch.randint(low=self._heading_change_steps_min, high=self._heading_change_steps_max,
                                     size=(n,), device=self.device, dtype=torch.int64)
        
        face_tar_dir = torch.stack([torch.cos(rand_face_theta), torch.sin(rand_face_theta)], dim=-1)

        self._tar_speed[env_ids] = tar_speed
        self._tar_dir[env_ids] = tar_dir                                                    # target root direction
        self._tar_facing_dir[env_ids] = face_tar_dir                                        # target root facing direction
        self._heading_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps
        return
    
    def _compute_task_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._humanoid_root_states
            tar_dir = self._tar_dir
            tar_speed = self._tar_speed
            tar_face_dir = self._tar_facing_dir
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_dir = self._tar_dir[env_ids]
            tar_speed = self._tar_speed[env_ids]
            tar_face_dir = self._tar_facing_dir[env_ids]
        
        obs = compute_heading_observations(root_states, tar_dir, tar_speed, tar_face_dir)
        return obs

    def _compute_reward(self, actions):
        # 1. imitation reward
        obs = self.obs_buf              # shape: [num_envs, 76]
        ref_obs = self.ref_buf           # shape: [num_envs, 88 or 91]
        key_pos = self._key_body_pos
        _print = True if(self.sim_forward or self.sim_forward_continuous) else False
        _step = self.progress_buf[0]
        imit_reward = compute_deepmm_reward(obs, ref_obs, key_pos, self.useCoM, self.useRootRot, len(self._dof_offsets)-1, _print, _step)

        # 2. heading task reward
        root_pos = self._humanoid_root_states[..., 0:3]
        root_rot = self._humanoid_root_states[..., 3:7]

        task_reward = compute_heading_reward(root_pos, self._prev_root_pos, root_rot,
                                                 self._tar_dir, self._tar_speed,
                                                 self._tar_facing_dir, self.dt)
        
        self.rew_buf[:] = imit_reward + task_reward
        
        return

    def _draw_task(self):
        self._update_marker()

        vel_scale = 0.2
        heading_cols = np.array([[0.0, 1.0, 0.0],
                                [1.0, 0.0, 0.0]], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        root_pos = self._humanoid_root_states[..., 0:3]
        prev_root_pos = self._prev_root_pos
        sim_vel = (root_pos - prev_root_pos) / self.dt
        sim_vel[..., -1] = 0

        starts = root_pos
        tar_ends = torch.clone(starts)
        tar_ends[..., 0:2] += vel_scale * self._tar_speed.unsqueeze(-1) * self._tar_dir
        sim_ends = starts + vel_scale * sim_vel

        verts = torch.cat([starts, tar_ends, starts, sim_ends], dim=-1).cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts[i:i+1]
            curr_verts = curr_verts.reshape([2, 6])
            self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, heading_cols)

        return
    
#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_heading_observations(root_states, tar_dir, tar_speed, tar_face_dir):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor
    root_rot = root_states[:, 3:7]

    tar_dir3d = torch.cat([tar_dir, torch.zeros_like(tar_dir[..., 0:1])], dim=-1)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    local_tar_dir = quat_rotate(heading_rot, tar_dir3d)
    local_tar_dir = local_tar_dir[..., 0:2]
    tar_speed = tar_speed.unsqueeze(-1)
    
    tar_face_dir3d = torch.cat([tar_face_dir, torch.zeros_like(tar_face_dir[..., 0:1])], dim=-1)
    local_tar_face_dir = quat_rotate(heading_rot, tar_face_dir3d)
    local_tar_face_dir = local_tar_face_dir[..., 0:2]

    obs = torch.cat([local_tar_dir, tar_speed, local_tar_face_dir], dim=-1)
    return obs

@torch.jit.script
def compute_heading_reward(root_pos, prev_root_pos, root_rot, tar_dir, tar_speed, tar_face_dir, dt, _print, _step):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, int) -> Tensor
    vel_err_scale = 0.25
    tangent_err_w = 0.1

    dir_reward_w = 0.7
    facing_reward_w = 0.3

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)

    tar_dir_vel = tar_dir_speed.unsqueeze(-1) * tar_dir
    tangent_vel = root_vel[..., :2] - tar_dir_vel

    tangent_speed = torch.sum(tangent_vel, dim=-1)

    tar_vel_err = tar_speed - tar_dir_speed
    tangent_vel_err = tangent_speed
    dir_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err + 
                        tangent_err_w * tangent_vel_err * tangent_vel_err))

    speed_mask = tar_dir_speed <= 0
    dir_reward[speed_mask] = 0

    heading_rot = torch_utils.calc_heading_quat(root_rot)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    facing_dir = quat_rotate(heading_rot, facing_dir)
    facing_err = torch.sum(tar_face_dir * facing_dir[..., 0:2], dim=-1)
    facing_reward = torch.clamp_min(facing_err, 0.0)

    reward = dir_reward_w * dir_reward + facing_reward_w * facing_reward

    if (_print):
        print("facing_reward: ", facing_reward[0].item(), " | ", (facing_reward_w * facing_reward)[0].item())
        print("dir_reward: ", dir_reward[0].item(), " | ", (dir_reward_w * dir_reward)[0].item())
        print("control total_reward: ", reward[0].item())

    return reward