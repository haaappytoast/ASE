
import os
import json
import numpy as np

from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive

import collections

'''
Refer to jupyter notebook file "make_SkeletonState_npy_file" for more detail
'''

# #### 1. fbx import -> ybot_fbx ####
# # fbx_file = "data/unity/ArmatureSkinningUpdateTpose.fbx"
# # fbx_file = "data/personal/t_pose/ybot_vis@T-Pose.fbx"
# fbx_file = "data/unity/t_pose/red@T-Pose.fbx"
# # import fbx file - make sure to provide a valid joint name for root_joint
# motion = SkeletonMotion.from_fbx(
#     fbx_file_path=fbx_file,
#     # root_joint="Hips",
#     root_joint="mixamorig:Hips",
#     # fps=60
#     fps=30
# )
# motion.to_file("data/personal/t_pose/ybot_vis@T-pose.npy")

#### 2. "SkeletonMotion" -> "SkeletonState"
# SKELETON npy file
avatar_name = "red"
# path = "data/personal/t_pose/"
path = "data/unity/t_pose/"

# RAW npy file
# npy_file_path = os.path.abspath( path + avatar_name + "@T-pose.npy")
npy_file_path = os.path.abspath(path + avatar_name + "@T-pose.npy")
avatar_t_pose = np.load(npy_file_path, allow_pickle=True).item()
print("---------\n1. SkeletonMotion TPOSE of " + avatar_name + "\n")
print(avatar_t_pose.keys())       ## odict_keys(['rotation', 'root_translation', 'global_velocity', 'global_angular_velocity', 'skeleton_tree', 'is_local', 'fps', '__name__'])
print("----------\n")

# SkeletonState
state_tpose = collections.OrderedDict()
state_tpose['rotation'] = {'arr': avatar_t_pose['rotation']['arr'][0], \
                          'context' : avatar_t_pose['rotation']['context']}
state_tpose['root_translation'] = {'arr': avatar_t_pose['root_translation']['arr'][0], \
                                  'context' : avatar_t_pose['root_translation']['context']}

state_tpose['skeleton_tree'] = avatar_t_pose['skeleton_tree']
state_tpose['is_local'] = avatar_t_pose['is_local']
state_tpose['__name__'] = 'SkeletonState'
print("---------\n2. SkeletonState TPOSE of " + avatar_name + "\n")
print(state_tpose['__name__'])
print(state_tpose.keys())
print("----------\n\n")

# path = "/home/vml/deepmm_ws/ASE/ase/poselib/data/personal/t_pose/"
path = os.path.abspath(path)
np.save(path + "/" + avatar_name + "_tpose", state_tpose)
print("---------\nSAVED SkeletonState TPOSE of " + avatar_name + " in " + path + "\n----------")
