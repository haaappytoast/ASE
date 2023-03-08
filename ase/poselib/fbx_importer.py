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


import os
import json

from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive

# source fbx file path
# fbx_file = "data/01_01_cmu.fbx"
fbx_file = "data/ybot_walking.fbx"

# import fbx file - make sure to provide a valid joint name for root_joint
# motion = SkeletonMotion.from_fbx(
#     fbx_file_path=fbx_file,
#     root_joint="mixamorig:Hips",
#     # fps=60
#     fps=30
# )

# import npy file
import numpy as np
path = "/home/vml/deepmm_ws/ASE/ase/poselib/data/"
npy_file = "concat_humanoid_walk.npy"

# path = "/home/vml/deepmm_ws/isaac_ws/isaac_example/ASE/ase/data/motions/"
# npy_file = "amp_humanoid_walk.npy"

motion_dict = np.load(path + npy_file, allow_pickle=True).item()

npy_motion = SkeletonMotion.from_dict(
    dict_repr=motion_dict
)


# # save motion in npy format
# motion.to_file("data/ybot_walking.npy")

# visualize motion
# plot_skeleton_motion_interactive(motion)
plot_skeleton_motion_interactive(npy_motion)
