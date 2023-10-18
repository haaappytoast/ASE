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


import torch

from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
from poselib.visualization.common import plot_skeleton_state

"""
This scripts imports a MJCF XML file and converts the skeleton into a SkeletonTree format.
It then generates a zero rotation pose, and adjusts the pose into a T-Pose.
"""

# # import MJCF file
# xml_path = "/home/vml/deepmm_ws/ASE/ase/data/assets/mjcf/cml_humanoid.xml"
# skeleton = SkeletonTree.from_mjcf(xml_path)

# # generate zero rotation pose
# zero_pose = SkeletonState.zero_pose(skeleton)

# adjust pose into a T Pose

# local_rotation = skeletonState.local_rotation

# local_rotation[skeletonTree.index("Hips")] = quat_mul(
#     quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True), 
#     local_rotation[skeletonTree.index("Hips")]
# )
# local_rotation[skeletonTree.index("Hips")] = quat_mul(
#     quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([0.0, 1.0, 0.0]), degree=True), 
#     local_rotation[skeletonTree.index("Hips")]
# )

# translation = zero_pose.root_translation
# translation += torch.tensor([0, 0, 0.9])
# translation = zero_pose.root_translation
# print("zero_pose.global_translation: ", zero_pose.global_translation)

# # save and visualize T-pose
# zero_pose.to_file("data/cml_humanoid_tpose110.npy")
# plot_skeleton_state(zero_pose)


# # cml npy file
path = "/home/njh/Works/yerim/ASE/ase/poselib/data/"
import numpy as np
npy_file = "cml_humanoid_tpose.npy"
t_pose = np.load(path + npy_file, allow_pickle=True).item()
skeletonState = SkeletonState.from_dict(t_pose) # skeletonState
skeletonTree = SkeletonTree.from_dict(t_pose["skeleton_tree"]) # skeletonState
print(t_pose)

print("---before---\n", skeletonState.root_translation)

# # # translate root translation to new pose
# # new_root_translation = torch.tensor([0, skeletonState.root_translation[1], 0])
# # new_pose = SkeletonState.from_rotation_and_root_translation(
# #             skeleton_tree=skeletonTree,
# #             r=skeletonState.local_rotation,
# #             t=new_root_translation,
# #             is_local=True
# #             )
# # translation = skeletonState.root_translation
# # translation = torch.tensor([0, translation[1], 0])
# # print("\n---after---\n", new_pose.root_translation)
# # new_pose.to_file(path + "ybot_new_tpose.npy")

plot_skeleton_state(skeletonState)