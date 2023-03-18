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

from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from rl_games.common import schedulers
from rl_games.common import vecenv

from isaacgym.torch_utils import *

import time
from datetime import datetime
import numpy as np
from torch import optim
import torch 
from torch import nn

import learning.replay_buffer as replay_buffer
import learning.common_agent as common_agent 

from tensorboardX import SummaryWriter

class DeepmmAgent(common_agent.CommonAgent):
    def __init__(self, base_name, config):
        #!! 이 안에서 [deepmm]_network_builder.py, [deepmm]_models.py 묶어줌.
        super().__init__(base_name, config)
        return

    #! experience해주는 코드 -> same as common_agent.py
    def play_steps(self):
        batch_dict = super().play_steps()
        return batch_dict
    
    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        return

    def train_epoch(self):
        train_info = super().train_epoch()
        return train_info


    def calc_gradients(self, input_dict):
        super().calc_gradients(input_dict=input_dict)
        return

    def _load_config_params(self, config):
        super()._load_config_params(config)
        self._task_reward_w = config['task_reward_w']
        return

    def _build_net_config(self):
        config = super()._build_net_config()
        return config
    
    def _init_train(self):
        super()._init_train()   # 아무것도 안함
        return