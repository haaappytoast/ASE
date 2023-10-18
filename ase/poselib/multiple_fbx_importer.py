import os
import json

from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
import yaml

#### fbx import -> ybot_fbx ####
# fbx_file = "data/unity/ArmatureSkinningUpdateTpose.fbx"

motion_path = "1018_outblock.yaml"
fbx_path = "data/unity/motion_retarget/1018_outblock/"
VISUALIZE = False

with open(motion_path) as f:
    file = yaml.full_load(f)

for data in file['motions']:
    fbx_motion = data['fbx']
    print(fbx_motion)


    # import fbx file - make sure to provide a valid joint name for root_joint
    motion = SkeletonMotion.from_fbx(
        fbx_file_path=fbx_path + fbx_motion,
        # root_joint="Hips",
        root_joint="mixamorig:Hips",
        fps=30
    )
    motion.to_file(fbx_path + fbx_motion[:-4] + ".npy")

    # visualize motion
    if VISUALIZE:
        plot_skeleton_motion_interactive(motion)