import os
import json

from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
import yaml

#### fbx import -> ybot_fbx ####
# fbx_file = "data/unity/ArmatureSkinningUpdateTpose.fbx"

motion_path = "1020_throw.yaml"
fbx_path = "data/unity/1020_throw/"
# motion_path = "1116_pickup.yaml"
# fbx_path = "data/unity/1116_ybot_pick/"
VISUALIZE = True
POSTPROCESS_FPS = True
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
    # # import fbx file - make sure to provide a valid joint name for root_joint
    # motion = SkeletonMotion.from_fbx(
    #     fbx_file_path=fbx_path + fbx_motion,
    #     # root_joint="Hips",
    #     root_joint="Hip",
    #     fps=30
    # )

    motion.to_file(fbx_path + fbx_motion[:-4] + ".npy")
    print(fbx_path + fbx_motion[:-4] + ".npy")
    if POSTPROCESS_FPS:
        import numpy as np
        motion_dict = np.load(fbx_path + fbx_motion[:-4] + ".npy", allow_pickle=True).item()
        motion_dict['fps'] = 30
        print("changed fps: ", motion_dict['fps'])

    # visualize motion
    if VISUALIZE:
        plot_skeleton_motion_interactive(motion)