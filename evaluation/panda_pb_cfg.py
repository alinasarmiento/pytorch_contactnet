from yacs.config import CfgNode as CN
import numpy as np

_C = CN()

# _C.SUBGOAL_TIMEOUT = 20
# _C.TIMEOUT = 20
_C.SUBGOAL_TIMEOUT = 25
_C.TIMEOUT = 50

_C.OBJECT_INIT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
_C.OBJECT_FINAL = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

_C.PALM_RIGHT = [0.0, -0.08, 0.025, 1.0, 0.0, 0.0, 0.0]
_C.PALM_LEFT = [-0.0672, -0.250, 0.205, 0.955, 0.106, 0.275, 0.0]

#_C.TIP_TO_WRIST_TF = [0.0, 0.071399, -0.14344421, 0.0, 0.0, 0.0, 1.0]
_C.TIP_TO_WRIST_TF = [0, 0, 0, 0, 0, 0, 1]
## _C.WRIST_TO_TIP_TF = [0.0, -0.071399, 0.14344421, 0.0, 0.0, 0.0, 1.0]
#_C.WRIST_TO_TIP_TF = [0.0, -0.0714, 0.15, 0.0, 0.0, 0.0, 1.0]
_C.WRIST_TO_TIP_TF = [0, 0, 0, 0, 0, 0, 1]
# starter poses for the robot, default is robot home position
_C.RIGHT_INIT = [0.413, -1.325, -1.040, -0.053, -0.484, 0.841, -1.546]
_C.LEFT_INIT = [-0.473, -1.450, 1.091, 0.031, 0.513, 0.77, -1.669]
_C.PANDA_INIT = [0, 0, 0, -1.2, 0, 1, -1]

_C.OBJECT_WORLD_XY = [0.3, 0.0]
_C.TABLE_HEIGHT = 0.00
_C.DELTA_Z = 0.000

_C.CAMERA_FOCUS = [0.4, 0.0, 0.1]
_C.OBJECT_POSE_1 = [0.3, 0.0, 0.0275, 0.0, 0.0, 0.0, 1.0]
_C.OBJECT_POSE_2 = [0.3, 0.0, 0.0275, 0.7071067811865475, 0.0, 0.0, 0.7071067811865476]
_C.OBJECT_POSE_3 = [0.3, 0.0, 0.0275, 0.0, 0.7071067811865475, 0.0, 0.7071067811865476]

####

_C.RIGHT_GEL_ID = 12
_C.LEFT_GEL_ID = 25
_C.TABLE_ID = 27

# contactDamping = alpha*contactStiffness
_C.ALPHA = 0.01
_C.GEL_CONTACT_STIFFNESS = 500
_C.GEL_RESTITUION = 0.99
_C.GEL_LATERAL_FRICTION = 1.0

_C.X_BOUNDS = [0.5, 0.55]
_C.Y_BOUNDS = [-0.05, 0.05]
_C.YAW_BOUNDS = [-np.pi/6, np.pi/6]
_C.DEFAULT_Z = 0.0
_C.DEFAULT_XY_POS = [0.4, 0.0]

_C.NUM_GRASP_SAMPLES = 1
_C.GRASP_DIST_TOLERANCE = 0.02

# palm y normal should be oriented within this boundary
_C.GRASP_MIN_Y_PALM_DEG = 45
_C.GRASP_MAX_Y_PALM_DEG = 135

# pairs of face indices that are valid for sampling a one step grasp (analyzed offline after testing each pair)
_C.VALID_GRASP_PAIRS = [[1, 2, 3, 4],
                        [0, 3, 4, 5],
                        [0, 3, 4, 5],
                        [0, 1, 2, 5],
                        [0, 1, 2, 5],
                        [1, 2, 3, 4]]

_C.PULL_TO_GRASP = [5, 0, 3, 4, 2, 1]
_C.GRASP_TO_PULL = [1, 5, 4, 2, 3, 0]

_C.BODY_TABLE_TF = [0.11091, 0.0, 0.0, 0.0, 0.0, 0.7071045443232222, 0.7071090180427969]

# transform from palm_tip to palm_tip2 in palm_tip frame
_C.TIP_TIP2_TF = [0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 1.0]

# directory that has the saved grasp samples
_C.GRASP_SAMPLES_DIR = 'catkin_ws/src/rpo_planning/src/rpo_planning/data/grasp_samples'

# maximum allowable motion to consider this a stable pose
_C.STABLE_POS_ERR = 0.005 # 5mm
_C.STABLE_ORI_ERR = 0.08 # about 5 deg

def get_cfg_defaults():
    return _C.clone()
