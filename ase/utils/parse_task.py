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

#!! should be always changed
from env.tasks.humanoid_test import HumanoidTest
from env.tasks.humanoid_test_heading_control import HumanoidHeadingControl
from env.tasks.humanoid_test_heading import HumanoidHeading
from env.tasks.humanoid_view import HumanoidDeepmimic

from env.tasks.humanoid import Humanoid
from env.tasks.humanoid_view_motion import HumanoidViewMotion
#! added for deepmm
from env.tasks.vec_task_wrappers import VecTaskPythonWrapper, VecTaskDeepmmWrapper

from isaacgym import rlgpu

import json
import numpy as np


def warn_task_name():
    raise Exception(
        "Unrecognized task!\nTask should be one of: [BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, ShadowHandLSTM, ShadowHandFFOpenAI, ShadowHandFFOpenAITest, ShadowHandOpenAI, ShadowHandOpenAITest, Ingenuity]")

def parse_task(args, cfg, cfg_train, sim_params):
    #! args는 get_args()로부터 얻어올 수 있음. defined in ase>utils.config.py
    # create native task and pass custom config
    device_id = args.device_id
    rl_device = args.rl_device

    #! cfg에 seed 추가
    cfg["seed"] = cfg_train.get("seed", -1)
    #! cfg에 "env", "sim" 2개의 section 중 "env"만!
    cfg_task = cfg["env"]
    cfg_task["seed"] = cfg["seed"]

    try:
        #! 잘보면, Humanoid의 parameter와 같음. 대박!
        #! args.task => argument로 주어주는 task
        task = eval(args.task)(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            device_type=args.device,
            device_id=device_id,
            headless=args.headless)
    except NameError as e:
        print(e)
        warn_task_name()

    # dict.get(key, value): 해당 키가 dict에 없을 때 value값 할당하고 value값 가져오기        
    # Task: Humanoid인거지!
    env_name = cfg_train['params']['config']['env_name']
    if(env_name =='rlgpu'):
        env = VecTaskPythonWrapper(task, rl_device, cfg_train.get("clip_observations", np.inf), cfg_train.get("clip_actions", 1.0))
    elif (env_name == 'mimic'):
        #! added for deepmm
        env = VecTaskDeepmmWrapper(task, rl_device, cfg_train.get("clip_observations", np.inf), cfg_train.get("clip_actions", 1.0))

    return task, env
