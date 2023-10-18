import torch

from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
import numpy as np

path = "/home/njh/Works/yerim/ASE/ase/poselib/data/retargeted/"
npy_file = "cml@jointInfo1.npy"

motion_dict = np.load(path + npy_file, allow_pickle=True).item()
print(motion_dict["rotation"]['arr'].shape)
# skeletonMotion = SkeletonMotion.from_dict(
#     dict_repr=motion_dict
# )

# plot_skeleton_motion_interactive(skeletonMotion)

# print(t_pose)

# print("---before---\n", skeletonState.root_translation)