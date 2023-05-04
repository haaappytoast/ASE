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
import env.tasks.humanoid_test_heading as humanoid_test_heading
from env.tasks.humanoid_test import compute_deepmm_reward

from utils import gym_util
from utils.motion_lib import DeepMimicMotionLib
from isaacgym.torch_utils import *

from utils import torch_utils
import sys
from poselib.poselib.core import *

TAR_ACTOR_ID = 1
TAR_FACING_ACTOR_ID = 2

class HumanoidHeadingControl(humanoid_test_heading.HumanoidHeading):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        if (not self.headless):
            # Keyboard Inputs for heading direction control
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_W, "heading_forward")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_A, "heading_left")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_S, "heading_backward")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_D, "heading_right")
            
            # Keyboard Inputs for facing direction control
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_UP, "facing_forward")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_LEFT, "facing_left")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_DOWN, "facing_backward")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_RIGHT, "facing_right")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_R, "random_directions")

        # keyboard 관련 variables
        self.heading_inputs = torch.zeros(4, dtype=torch.bool)
        self.facing_inputs = torch.zeros(4, dtype=torch.bool)
        return  
    
    def render(self, sync_frame_time=False):
        super().render(sync_frame_time)

        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                forward_count = 15
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "sim_forward" and evt.value > 0:
                    print("\nSimulate 1 frame! ")
                    self.enable_viewer_sync = True
                    self.sim_forward = True
                    pass
                elif evt.action == "sim_pause" and evt.value > 0:
                    print("Resume/Pause simulation! ")
                    self.sim_pause = not self.sim_pause
                    pass
                elif evt.action == "sim_forward_continuous" and evt.value > 0:
                    print("Simulate", forward_count, "frames! ")
                    self.sim_forward_continuous = True
                    self.forward_count = forward_count
                    pass
                elif evt.action == "keyboard_help" and evt.value>0:
                    print("-----------------------")
                    print("Tab: Resume/Pause Simulation")
                    print("C: Step", forward_count, "frames ahead")
                    print(">: Step 1 frame ahead")
                    print("-----------------------")
                    pass

                elif evt.action == "heading_forward" and evt.value > 0:
                    idx = 0
                    self.heading_inputs = torch.zeros(4, dtype=torch.bool)
                    self.heading_inputs[idx] = True
                    pass
                elif evt.action == "heading_left" and evt.value > 0:
                    idx = 1
                    self.heading_inputs = torch.zeros(4, dtype=torch.bool)
                    self.heading_inputs[idx] = True
                    pass
                elif evt.action == "heading_backward" and evt.value > 0:
                    idx = 2
                    self.heading_inputs = torch.zeros(4, dtype=torch.bool)
                    self.heading_inputs[idx] = True
                    pass
                elif evt.action == "heading_right" and evt.value > 0:
                    idx = 3
                    self.heading_inputs = torch.zeros(4, dtype=torch.bool)
                    self.heading_inputs[idx] = True
                    pass
                elif evt.action == "facing_forward" and evt.value > 0:
                    idx = 0
                    self.facing_inputs = torch.zeros(4, dtype=torch.bool)
                    self.facing_inputs[idx] = True
                    pass
                elif evt.action == "facing_left" and evt.value > 0:
                    idx = 1
                    self.facing_inputs = torch.zeros(4, dtype=torch.bool)
                    self.facing_inputs[idx] = True
                    pass
                elif evt.action == "facing_backward" and evt.value > 0:
                    idx = 2
                    self.facing_inputs = torch.zeros(4, dtype=torch.bool)
                    self.facing_inputs[idx] = True
                    pass
                elif evt.action == "facing_right" and evt.value > 0:
                    idx = 3
                    self.facing_inputs = torch.zeros(4, dtype=torch.bool)
                    self.facing_inputs[idx] = True
                    pass
                # elif evt.action == "random_directions" and evt.value > 0:
                #     self.rand_inputs = torch.zeros(0, dtype=torch.bool)
                #     pass
        return

    def update_task_by_keyboard(self):
        # keyboard에 대한 값 체크하기
        heading_ids = self.heading_inputs.nonzero(as_tuple=False).flatten()
        heading_input = heading_ids.shape[0]

        facing_ids = self.facing_inputs.nonzero(as_tuple=False).flatten()
        facing_input = facing_ids.shape[0]

        n = self.num_envs

        # 1. keyboard가 heading_forward(W) 라면
        if (self.heading_inputs[0] == True):
            # tar_dir을 x_direction 값으로 (rand_theta = 0)
            rand_theta = torch.zeros(n, device=self.device)
            pass

        # 2. keyboard가 heading_left(A) 라면
        elif (self.heading_inputs[1] == True):
            # tar_dir을 x_direction 값으로 (rand_theta = 90)
            rand_theta = 1/2 * np.pi * torch.ones(n, device=self.device)
            pass

        # 3. keyboard가 heading_backward(S) 라면
        elif (self.heading_inputs[2] == True):
            # tar_dir을 x_direction 값으로 (rand_theta = 180)
            rand_theta = np.pi * torch.ones(n, device=self.device)
            pass
        
        # 4. keyboard가 heading_right(D) 라면
        elif (self.heading_inputs[3] == True):
            # tar_dir을 x_direction 값으로 (rand_theta = -90)
            rand_theta = -1/2 * np.pi * torch.ones(n, device=self.device)
            pass
        

        # 4. keyboard가 facing_forward(upArrow) 라면
        # tar_dir을 x_direction 값으로 (rand_face_theta = 0)
        if (self.facing_inputs[0] == True):
            rand_face_theta = torch.zeros(n, device=self.device)
            pass
        
        # 5. keyboard가 facing_left(lArrow) 라면
        # tar_dir을 x_direction 값으로 (rand_face_theta = 90)
        elif (self.facing_inputs[1] == True):
            rand_face_theta = 1/2 * np.pi * torch.ones(n, device=self.device)
            pass
        
        # 6. keyboard가 facing_backward(backArrow) 라면
        # tar_dir을 x_direction 값으로 (rand_face_theta = 180)
        elif (self.facing_inputs[2] == True):
            rand_face_theta = np.pi * torch.ones(n, device=self.device)
            pass

        # 7. keyboard가 facing_right(rArrow) 라면
        # tar_dir을 x_direction 값으로 (rand_face_theta = -90)
        elif (self.facing_inputs[3] == True):
            rand_face_theta = -1/2 * np.pi * torch.ones(n, device=self.device)
            pass

        # keyboard input값이 하나라도 있다면
        if (heading_input is not 0):
        # # rand_theta와 rand_face_theta를 tar_dir와 tar_facing_dir에 넣어주기
        # #! 고민! reset 되는 environment에 대해서만 해주는 것인가?! -> 이 부분도 내가 정해야 되는거군! 
            tar_dir = torch.stack([torch.cos(rand_theta), torch.sin(rand_theta)], dim=-1)
            self._tar_dir = tar_dir                                                         # target root direction
            tar_dir3d = torch.cat([tar_dir, torch.zeros_like(tar_dir[..., 0:1])], dim=-1)
            root_rot = self._humanoid_root_states[:, 3:7]

            # local x_dir of character in global coordinate
            heading_rot = torch_utils.calc_heading_quat(root_rot)
            tar_dir_wrt_root = quat_rotate(heading_rot, tar_dir3d)
            self._tar_dir = tar_dir_wrt_root[..., 0:2]
            heading_input = 0
            self.heading_inputs = torch.zeros(4, dtype=torch.bool)

        # keyboard input값이 하나라도 있다면
        if (facing_input is not 0):
        # # rand_theta와 rand_face_theta를 tar_dir와 tar_facing_dir에 넣어주기
        # #! 고민! reset 되는 environment에 대해서만 해주는 것인가?! -> 이 부분도 내가 정해야 되는거군! 
            face_tar_dir = torch.stack([torch.cos(rand_face_theta), torch.sin(rand_face_theta)], dim=-1)
            face_tar_dir3d = torch.cat([face_tar_dir, torch.zeros_like(face_tar_dir[..., 0:1])], dim=-1)
            root_rot = self._humanoid_root_states[:, 3:7]

            # local x_dir of character in global coordinate
            heading_rot = torch_utils.calc_heading_quat(root_rot)
            tar_facing_dir_wrt_root = quat_rotate(heading_rot, face_tar_dir3d)
            self._tar_facing_dir = tar_facing_dir_wrt_root[..., 0:2]                                     # target root facing direction
            facing_input = 0
            self.facing_inputs = torch.zeros(4, dtype=torch.bool)

        change_steps = torch.randint(low=self._heading_change_steps_min, high=self._heading_change_steps_max,
                                        size=(n,), device=self.device, dtype=torch.int64)
        self._heading_change_steps = self.progress_buf + change_steps
        print("self._heading_change_steps: ", self._heading_change_steps)
        return


    def rotate_heading_dir(self):
        pass
    
    def rotate_facing_dir(self):
        pass

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._update_task()
        self.update_task_by_keyboard()
        return