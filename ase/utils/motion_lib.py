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

import numpy as np
import os
import yaml

from poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
from poselib.poselib.core.rotation3d import *
from isaacgym.torch_utils import *

from utils import torch_utils

import torch

USE_CACHE = True
print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)

if not USE_CACHE:
    old_numpy = torch.Tensor.numpy
    class Patch:
        def numpy(self):
            if self.is_cuda:
                return self.to("cpu").numpy()
            else:
                return old_numpy(self)

    torch.Tensor.numpy = Patch.numpy

class DeviceCache:
    def __init__(self, obj, device):
        self.obj = obj
        self.device = device

        keys = dir(obj)
        num_added = 0
        for k in keys:
            try:
                out = getattr(obj, k)
            except:
                print("Error for key=", k)
                continue

            if isinstance(out, torch.Tensor):
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)  
                num_added += 1
            elif isinstance(out, np.ndarray):
                out = torch.tensor(out)
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)
                num_added += 1
        
        print("Total added", num_added)

    def __getattr__(self, string):
        out = getattr(self.obj, string)
        return out


class MotionLib():
    def __init__(self, motion_file, dof_body_ids, dof_offsets,
                 key_body_ids, device):
        self._dof_body_ids = dof_body_ids
        self._dof_offsets = dof_offsets
        self._num_dof = dof_offsets[-1] #! 28 for humanoid
        self._key_body_ids = torch.tensor(key_body_ids, device=device)
        self._device = device
        self._load_motions(motion_file)

        #! skeleton3d.py > SkeletonMotion
        motions = self._motions
        self.gts = torch.cat([m.global_translation for m in motions], dim=0).float()                # global translation:           [motion file들의 num_frames * num_motion_file, num_rigid_bodies, 3]
        self.grs = torch.cat([m.global_rotation for m in motions], dim=0).float()                   # global rotation:              [motion file들의 num_frames * num_motion_file, num_rigid_bodies, 4]
        self.lrs = torch.cat([m.local_rotation for m in motions], dim=0).float()                    # body local rotation:          [motion file들의 num_frames, num_rigid_bodies, 4]
        self.grvs = torch.cat([m.global_root_velocity for m in motions], dim=0).float()             # global root velocity:         [motion file들의 num_frames * num_motion_file, 3]
        self.gravs = torch.cat([m.global_root_angular_velocity for m in motions], dim=0).float()    # global_root_angular_velocity: [motion file들의 num_frames * num_motion_file, 4]
        #! from _compute_motion_dof_vels (local_angular_velocity from difference b/w local_rots)
        self.dvs = torch.cat([m.dof_vels for m in motions], dim=0).float()  # local dof joint velocity  # local_angular_velocity

        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)

        self.motion_ids = torch.arange(len(self._motions), dtype=torch.long, device=self._device)

        return

    def num_motions(self):
        return len(self._motions)

    def get_total_length(self):
        return sum(self._motion_lengths)

    def get_motion(self, motion_id):
        return self._motions[motion_id]

    def sample_motions(self, n):
        motion_ids = torch.multinomial(self._motion_weights, num_samples=n, replacement=True)

        # m = self.num_motions()
        # motion_ids = np.random.choice(m, size=n, replace=True, p=self._motion_weights)
        # motion_ids = torch.tensor(motion_ids, device=self._device, dtype=torch.long)
        return motion_ids

    def sample_time(self, motion_ids, truncate_time=None):
        n = len(motion_ids)
        phase = torch.rand(motion_ids.shape, device=self._device)   # shape: [num_samples]
        motion_len = self._motion_lengths[motion_ids]

        truncate_time = 3
        if (truncate_time is not None):
            assert(truncate_time >= 0.0)
            motion_len -= truncate_time
        motion_time = phase * motion_len    # shape: [num_samples]

        return motion_time

    def get_motion_length(self, motion_ids):
        return self._motion_lengths[motion_ids]

    def get_motion_state(self, motion_ids, motion_times):
        n = len(motion_ids)
        num_bodies = self._get_num_bodies()
        num_key_bodies = self._key_body_ids.shape[0]

        motion_len = self._motion_lengths[motion_ids]       
        num_frames = self._motion_num_frames[motion_ids]    
        dt = self._motion_dt[motion_ids]                    

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)

        # frame_idx0 = torch.tensor([0]).to(device=0)
        # frame_idx1 = torch.tensor([0]).to(device=0)
        # blend = torch.Tensor([0.0]).to(device=0)
        # print("after // frame_idx0", frame_idx0.item(),", frame_idx1: ", frame_idx1.item(), ",blend: ", blend.item())
        
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        root_pos0 = self.gts[f0l, 0]
        root_pos1 = self.gts[f1l, 0]

        root_rot0 = self.grs[f0l, 0]
        root_rot1 = self.grs[f1l, 0]
        
        local_rot0 = self.lrs[f0l]
        local_rot1 = self.lrs[f1l]

        root_vel = self.grvs[f0l]

        root_ang_vel = self.gravs[f0l]
        
        key_pos0 = self.gts[f0l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]
        key_pos1 = self.gts[f1l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]

        dof_vel = self.dvs[f0l]

        vals = [root_pos0, root_pos1, local_rot0, local_rot1, root_vel, root_ang_vel, key_pos0, key_pos1]
        for v in vals:
            assert v.dtype != torch.float64


        blend = blend.unsqueeze(-1)

        root_pos = (1.0 - blend) * root_pos0 + blend * root_pos1

        root_rot = torch_utils.slerp(root_rot0, root_rot1, blend)

        blend_exp = blend.unsqueeze(-1)
        key_pos = (1.0 - blend_exp) * key_pos0 + blend_exp * key_pos1
        
        local_rot = torch_utils.slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))      # [num_envs, 15, 4]
        
        dof_pos = self._local_rotation_to_dof(local_rot)

        return root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos
    
    def _load_motions(self, motion_file):
        self._motions = []
        self._motion_lengths = []
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_files = []

        total_len = 0.0

        motion_files, motion_weights = self._fetch_motion_files(motion_file)
        num_motion_files = len(motion_files)
        for f in range(num_motion_files):
            curr_file = motion_files[f]
            print("Loading {:d}/{:d} motion files: {:s}".format(f + 1, num_motion_files, curr_file))
            curr_motion = SkeletonMotion.from_file(curr_file)
            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps

            num_frames = curr_motion.tensor.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)

            self._motion_fps.append(motion_fps)
            self._motion_dt.append(curr_dt)
            self._motion_num_frames.append(num_frames)
 
            curr_dof_vels = self._compute_motion_dof_vels(curr_motion)
            curr_motion.dof_vels = curr_dof_vels

            # Moving motion tensors to the GPU
            if USE_CACHE:
                curr_motion = DeviceCache(curr_motion, self._device)                
            else:
                curr_motion.tensor = curr_motion.tensor.to(self._device)
                curr_motion._skeleton_tree._parent_indices = curr_motion._skeleton_tree._parent_indices.to(self._device)
                curr_motion._skeleton_tree._local_translation = curr_motion._skeleton_tree._local_translation.to(self._device)
                curr_motion._rotation = curr_motion._rotation.to(self._device)

            self._motions.append(curr_motion)
            self._motion_lengths.append(curr_len)
            
            curr_weight = motion_weights[f]
            self._motion_weights.append(curr_weight)
            self._motion_files.append(curr_file)

        self._motion_lengths = torch.tensor(self._motion_lengths, device=self._device, dtype=torch.float32)

        self._motion_weights = torch.tensor(self._motion_weights, dtype=torch.float32, device=self._device)
        self._motion_weights /= self._motion_weights.sum()

        self._motion_fps = torch.tensor(self._motion_fps, device=self._device, dtype=torch.float32)
        self._motion_dt = torch.tensor(self._motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, device=self._device)


        num_motions = self.num_motions()
        total_len = self.get_total_length()

        print("Loaded {:d} motions with a total length of {:.3f}s.".format(num_motions, total_len))

        return

    def _fetch_motion_files(self, motion_file):
        ext = os.path.splitext(motion_file)[1]
        if (ext == ".yaml"):
            dir_name = os.path.dirname(motion_file)
            motion_files = []
            motion_weights = []

            with open(os.path.join(os.getcwd(), motion_file), 'r') as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)

            motion_list = motion_config['motions']
            for motion_entry in motion_list:
                curr_file = motion_entry['file']
                curr_weight = motion_entry['weight']
                assert(curr_weight >= 0)

                curr_file = os.path.join(dir_name, curr_file)
                motion_weights.append(curr_weight)
                motion_files.append(curr_file)
        else:
            motion_files = [motion_file]
            motion_weights = [1.0]

        return motion_files, motion_weights

    def _calc_frame_blend(self, time, len, num_frames, dt):
        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = (time - frame_idx0 * dt) / dt

        # print("time: ", time.item())
        # print("before // frame_idx0", frame_idx0.item(),", frame_idx1: ", frame_idx1.item(), ",blend: ", blend.item())

        return frame_idx0, frame_idx1, blend

    def _get_num_bodies(self):
        motion = self.get_motion(0)
        num_bodies = motion.num_joints
        return num_bodies

    def _compute_motion_dof_vels(self, motion):
        num_frames = motion.tensor.shape[0]
        dt = 1.0 / motion.fps
        dof_vels = []

        for f in range(num_frames - 1):
            local_rot0 = motion.local_rotation[f]   #? [15, 4] for humanoid -> yes
            local_rot1 = motion.local_rotation[f + 1]
            frame_dof_vel = self._local_rotation_to_dof_vel(local_rot0, local_rot1, dt)
            frame_dof_vel = frame_dof_vel
            dof_vels.append(frame_dof_vel)
        
        dof_vels.append(dof_vels[-1])
        dof_vels = torch.stack(dof_vels, dim=0)

        return dof_vels
    
    def _local_rotation_to_dof(self, local_rot):
        body_ids = self._dof_body_ids
        dof_offsets = self._dof_offsets

        n = local_rot.shape[0]
        dof_pos = torch.zeros((n, self._num_dof), dtype=torch.float, device=self._device)

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if (joint_size == 3):
                joint_q = local_rot[:, body_id]                         # body의 local rot에서 해당하는 joint의 local rot 가져오기 / size: [1, 4]
                joint_exp_map = torch_utils.quat_to_exp_map(joint_q)    
                dof_pos[:, joint_offset:(joint_offset + joint_size)] = joint_exp_map
            elif (joint_size == 1):
                joint_q = local_rot[:, body_id]
                joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                joint_theta = joint_theta * joint_axis[..., 1] # assume joint is always along y axis

                joint_theta = normalize_angle(joint_theta)
                dof_pos[:, joint_offset] = joint_theta

            else:
                print("Unsupported joint type")
                assert(False)

        return dof_pos

    def _local_rotation_to_dof_vel(self, local_rot0, local_rot1, dt):
        body_ids = self._dof_body_ids            # [1,  ~ ,14]
        dof_offsets = self._dof_offsets          # [0, 3, 6 , ~ , 28]
        #! joint의 axis 마다 다 vel 를 구하는 구나!
        dof_vel = torch.zeros([self._num_dof], device=self._device)

        diff_quat_data = quat_mul_norm(quat_inverse(local_rot0), local_rot1)
        diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
        local_vel = diff_axis * diff_angle.unsqueeze(-1) / dt   #! theta * (x_axis, y_axis, z_axis)
        local_vel = local_vel

        for j in range(len(body_ids)):  # 14개
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset  #! 그래서 이렇게 마지막 joint_offset에 28까지 넣어준거구만

            if (joint_size == 3):
                joint_vel = local_vel[body_id]  #! local_vel은 body의 local velocity 임.
                dof_vel[joint_offset:(joint_offset + joint_size)] = joint_vel #! 이걸 맞는 dof_vel에 axis에 맞게 (joint에 주는 거임)

            elif (joint_size == 1):
                assert(joint_size == 1)
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset] = joint_vel[1] # assume joint is always along y axis

            else:
                print("Unsupported joint type")
                assert(False)

        return dof_vel
    
    
class DeepMimicMotionLib(MotionLib):
    def __init__(self, motion_file, dof_body_ids, dof_offsets, key_body_ids, device):
        super().__init__(motion_file, dof_body_ids, dof_offsets, key_body_ids, device)
        motions = self._motions
        self.gvs = torch.cat([m.global_velocity for m in motions], dim=0).float()
        self.gavs = torch.cat([m.global_angular_velocity for m in motions], dim=0).float()
    
    def sample_time(self, motion_ids, max_episode_length, dt=None, train_epoch=None, is_train=True):

        n = len(motion_ids)
        phase = torch.rand(motion_ids.shape, device=self._device)   # shape: [num_samples]
        motion_len = self._motion_lengths[motion_ids]
        motion_num_frames = self._motion_num_frames[motion_ids]

        # resampling motion_time for env which is over max_episode_length
        boundary = (motion_num_frames - max_episode_length) * dt
        motion_time = phase * motion_len    # shape: [num_samples]
        
        overred = boundary < motion_time
        env_overred = torch.where(overred == True)


        # if (phase[env_overred].shape[0] == 0):
        #     pass
        # else:
        #     new_motion_time = torch.mul(phase[env_overred], boundary[env_overred])
        #     new_motion_time = boundary
        #     motion_time[env_overred] = new_motion_time

        return motion_time

    def _calc_phase(self, motion_ids, motion_times):
        motion_len = self._motion_lengths[motion_ids]       
        phase = motion_times/motion_len - motion_times // motion_len
        
        phase = torch.clip(phase, 0.0, 1.0)

        return phase
    
        

        return body_orn, body_ang_vel
    
    def _get_body_global_quat(self, motion_ids, motion_times):

        motion_len = self._motion_lengths[motion_ids]       
        num_frames = self._motion_num_frames[motion_ids]    
        dt = self._motion_dt[motion_ids]                    

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)

        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]
        
        global_quat0 = self.grs[f0l]
        global_quat1 = self.grs[f1l]

        vals = [global_quat0, global_quat1]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        blended_global_quat = torch_utils.slerp(global_quat0, global_quat1, blend)

        return blended_global_quat

    def _get_dof_local_quat(self, motion_ids, motion_times):

        motion_len = self._motion_lengths[motion_ids]       
        num_frames = self._motion_num_frames[motion_ids]    
        dt = self._motion_dt[motion_ids]                    

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        
        local_rot0 = self.lrs[f0l]  # (num_envs, rigid_body_size, 4)
        local_rot1 = self.lrs[f1l]

        vals = [local_rot0, local_rot1]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)
        # if (blend.shape[0] > 1):g
        #     blend = blend.unsqueeze(-2)
        #     blend = blend.repeat((1, local_rot0.shape[1], 1))   # shape: [2, 15, 1]

        local_dof = []
        for i in range(local_rot0.shape[0]):
            blended_local_rot = torch_utils.slerp(local_rot0[i, :, :].unsqueeze(0), local_rot1[i, :, :].unsqueeze(0), blend[i, :].unsqueeze(0))
            dof_pos = self._local_rotation_to_dof(blended_local_rot)
            local_dof.append(dof_pos)
        
        blended_local_rots = torch.vstack(local_dof)
        return blended_local_rots    # [1, num_joints * 4]
    
    def _get_jnt_local_angvel(self, motion_ids, motion_times):
        motion_len = self._motion_lengths[motion_ids]       
        num_frames = self._motion_num_frames[motion_ids]    
        dt = self._motion_dt[motion_ids]                    

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)

        f0l = frame_idx0 + self.length_starts[motion_ids]
        
        dof_vel = self.dvs[f0l]

        return dof_vel    # [1, num_dof]
    
    def _get_body_local_angvel(self, motion_ids, motion_times):
        motion_len = self._motion_lengths[motion_ids]       
        num_frames = self._motion_num_frames[motion_ids]    
        dt = self._motion_dt[motion_ids]                    

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)

        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        # root 관련 정보
        root_rot0 = self.grs[f0l, 0]
        root_rot1 = self.grs[f1l, 0]

        # global body angular velocity 관련 정보 
        body_global_ang_vel0 = self.gavs[f0l]
        body_global_ang_vel1 = self.gavs[f1l]

        vals = [root_rot0, root_rot1, body_global_ang_vel0, body_global_ang_vel1]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        if (blend.shape[0] > 1):
            blend_expand = blend.unsqueeze(-2)
            blend_expand = blend_expand.repeat((1, body_global_ang_vel0.shape[1], 1))   # shape: [2, 15, 1]
            body_glob_ang_vel = torch_utils.slerp(body_global_ang_vel0, body_global_ang_vel1, blend_expand)   # [2, 15, 3]
        else:
            body_glob_ang_vel = torch_utils.slerp(body_global_ang_vel0, body_global_ang_vel1, blend)   # [1, 15, 3]

        # root blending
        root_rot = torch_utils.slerp(root_rot0, root_rot1, blend)
        # global body angular velocity blending
        flat_body_ang_vel = body_glob_ang_vel.reshape(body_glob_ang_vel.shape[0] * body_glob_ang_vel.shape[1], 
                                                            body_glob_ang_vel.shape[2])                         # [num_env * rigid_body_size, 3]
        

        # heading_rot 구하기
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)   # quat from heading to ref_dir(global x-axis)   [num_envs, 4]
        heading_rot_expand = heading_rot.unsqueeze(-2)              # [1, 1, 4]
        heading_rot_expand = heading_rot_expand.repeat((1, body_glob_ang_vel.shape[1], 1))      # [1, 15, 4]
        flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                                        heading_rot_expand.shape[2])
        
        # calculate local body angular velocity
        local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel) #! local(root coordinate)에서 바라본 body rot / shape: [15 * num_envs, 4]
        local_body_ang_vel = local_body_ang_vel.reshape(heading_rot.shape[0], -1, local_body_ang_vel.shape[-1])

        return local_body_ang_vel    # [num_envs, 15, 3]
        
    def _get_ee_world_position(self, motion_ids, motion_times):
        motion_len = self._motion_lengths[motion_ids]       
        num_frames = self._motion_num_frames[motion_ids]    
        dt = self._motion_dt[motion_ids]                    

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)

        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]
        
        key_pos0 = self.gts[f0l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]
        key_pos1 = self.gts[f1l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]

        vals = [key_pos0, key_pos1]
        for v in vals:
            assert v.dtype != torch.float64

        blend_exp = blend.unsqueeze(-1)
        if (blend.shape[0] > 1):
            blend_exp_expand = blend_exp.unsqueeze(-1)
            key_pos = (1.0 - blend_exp_expand) * key_pos0 + blend_exp_expand * key_pos1
        else:
            key_pos = (1.0 - blend_exp) * key_pos0 + blend_exp * key_pos1
        
        return key_pos # [1, num_key_bodies, 3]

    def get_motion_state_for_reference(self, motion_ids, motion_times):
        local_body_rot = self._get_dof_local_quat(motion_ids, motion_times)
        local_body_angvel = self._get_body_local_angvel(motion_ids, motion_times)
        global_ee_pos = self._get_ee_world_position(motion_ids, motion_times)
        return local_body_rot, local_body_angvel, global_ee_pos

    def _get_blended_global_rot(self, motion_ids, motion_times):
        n = len(motion_ids)
        num_bodies = self._get_num_bodies()
        num_key_bodies = self._key_body_ids.shape[0]

        motion_len = self._motion_lengths[motion_ids]       
        num_frames = self._motion_num_frames[motion_ids]    
        dt = self._motion_dt[motion_ids]                    

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        # frame_idx0 = torch.tensor([0]).to(device=0)
        # frame_idx1 = torch.tensor([0]).to(device=0)
        # blend = torch.Tensor([0.0]).to(device=0)
        # print("after // frame_idx0", frame_idx0.item(),", frame_idx1: ", frame_idx1.item(), ",blend: ", blend.item())
        
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        global_rot0 = self.grs[f0l]
        global_rot1 = self.grs[f1l]

        
        vals = [global_rot0, global_rot1]
        for v in vals:
            assert v.dtype != torch.float64


        blend = blend.unsqueeze(-1)


        blend_exp = blend.unsqueeze(-1)
        
        global_rot = torch_utils.slerp(global_rot0, global_rot1, torch.unsqueeze(blend, axis=-1))
        

        return global_rot
    
    def _get_blended_local_rot(self, motion_ids, motion_times):
        n = len(motion_ids)
        num_bodies = self._get_num_bodies()
        num_key_bodies = self._key_body_ids.shape[0]

        motion_len = self._motion_lengths[motion_ids]       
        num_frames = self._motion_num_frames[motion_ids]    
        dt = self._motion_dt[motion_ids]                    

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        # frame_idx0 = torch.tensor([0]).to(device=0)
        # frame_idx1 = torch.tensor([0]).to(device=0)
        # blend = torch.Tensor([0.0]).to(device=0)
        # print("after // frame_idx0", frame_idx0.item(),", frame_idx1: ", frame_idx1.item(), ",blend: ", blend.item())
        
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        
        local_rot0 = self.lrs[f0l]
        local_rot1 = self.lrs[f1l]

        
        vals = [local_rot0, local_rot1]
        for v in vals:
            assert v.dtype != torch.float64


        blend = blend.unsqueeze(-1)
        
        local_rot = torch_utils.slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
        

        return local_rot
    