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
import numpy as np
import torch
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
from poselib.core.rotation3d import euclidean_to_transform, transform_rotation, transform_translation
from typing import Type

#! 1. from txt file to Transformation Matrix of Joint Rotation of motion file
def from_txt_to_npy(path, txt_name, save_np, np_name, fps=30):
    rot_file = open(path + txt_name)
    rot_json = json.load(rot_file)

    # parse motion info
    frameCount = rot_json['frameCount']
    jointCount = rot_json['jointCount']
    jnt_trans = rot_json['jointTrans']
    jnt_names = rot_json['jointNames']
    parentIdx = rot_json['parentIdx']    

    # changejnt_trans to numpy
    list_rot = []
    for i in range(0, len(jnt_trans)):
        list_rot.append(np.fromstring(jnt_trans[i], sep=' '))

    print("=========")
    print("frameCount:", frameCount)
    print("jointCount:", jointCount)
    print("\njnt_names:", jnt_names)
    print("\nparentIdx:", parentIdx)
    print("=========")
    
    rot_mat = np.array(list_rot)
    rot_mat = rot_mat.reshape(frameCount, jointCount, 4, 4)

    if save_np:
        print("--------------- save Transformation Matrix npy file: ", np_name, "--------------")
        np.save(path + np_name, rot_mat)

    return jnt_names, parentIdx, rot_mat, fps

def txt_to_motion(
        path,
        txt_name,
        save_np = True,
        np_name = 'unityCharac@motion@jointTransMat',
        skeleton_tree=None, 
        is_local=True,
        root_trans_index=0):
    
    joint_names, joint_parents, transforms, fps = from_txt_to_npy(path, txt_name, save_np, np_name, fps=30)

    local_transform = euclidean_to_transform(
    transformation_matrix=torch.from_numpy(transforms).float()
    )

    local_rotation = transform_rotation(local_transform)
    root_translation = transform_translation(local_transform)[..., root_trans_index, :]
    joint_parents = torch.from_numpy(np.array(joint_parents)).int()

    if skeleton_tree is None:
        local_translation = transform_translation(local_transform).reshape(
            -1, len(joint_parents), 3
        )[0]
        skeleton_tree = SkeletonTree(joint_names, joint_parents, local_translation)
    skeleton_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree, r=local_rotation, t=root_translation, is_local=True
    )
    if not is_local:
        skeleton_state = skeleton_state.global_repr()
    
    
    return SkeletonMotion.from_skeleton_state(
        skeleton_state=skeleton_state, fps=fps
    )


def main():
    path = 'data/unity/'
    # txt_name = 'motion_retarget/1018_user/user@jointInfo1.txt'
    # txt_name = 'test/1023_pickfruits/1023_picking_fruits_motion.txt'
    # txt_name = 'test/1024_throw/meta@throwjointInfo0 (9).txt'
    # txt_name = '1127_tennis/tennis_1127_TEST0.txt'
    # txt_name = '1127_tennis/user_TEST50s (3).txt'
    # txt_name = '1122_punch/meta@user_punchTEST50s (1).txt'
    # txt_name = '_user_motions/meta@1212punch2_50s.txt'
    txt_name = '1017_blocking/meta@block50s.txt'
    motion = txt_to_motion(path, txt_name, save_np = False, np_name='CMLAvatar@TempMotion@Transmat')

    motion.to_file("data/unity/" + txt_name[:-4]+".npy")

    # visualize motion
    plot_skeleton_motion_interactive(motion)


if __name__ == '__main__':
    main()
