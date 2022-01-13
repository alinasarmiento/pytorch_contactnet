import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

class Position:
    def __init__(self):
        self.x = 0.
        self.y = 0.
        self.z = 0.


class Orientation:
    def __init__(self):
        self.x = 0.
        self.y = 0.
        self.z = 0.
        self.w = 0.


class Pose:
    def __init__(self, position, orientation):
        self.position = position
        self.orientation = orientation


class Header:
    def __init__(self):
        self.frame_id = "world"


class PoseStamped():
    def __init__(self):
        position = Position()
        orientation = Orientation()
        pose = Pose(position, orientation)
        header = Header()
        self.pose = pose
        self.header = header

def pose_from_matrix(matrix, frame_id="world"):
    quat = R.from_dcm(matrix[:3, :3]).as_quat()
    trans = matrix[:-1, -1]
    pose = list(trans) + list(quat)
    pose = list2pose_stamped(pose, frame_id=frame_id)
    return pose
        
def list2pose_stamped(pose, frame_id="world"):
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.pose.position.x = pose[0]
    msg.pose.position.y = pose[1]
    msg.pose.position.z = pose[2]
    msg.pose.orientation.x = pose[3]
    msg.pose.orientation.y = pose[4]
    msg.pose.orientation.z = pose[5]
    msg.pose.orientation.w = pose[6]
    return msg


def unit_pose():
    return list2pose_stamped([0, 0, 0, 0, 0, 0, 1])


def convert_reference_frame(pose_source, pose_frame_target, pose_frame_source, frame_id=None):
    T_pose_source = matrix_from_pose(pose_source)
    pose_transform_target2source = get_transform(
        pose_frame_source, pose_frame_target)
    T_pose_transform_target2source = matrix_from_pose(
        pose_transform_target2source)
    T_pose_target = np.matmul(T_pose_transform_target2source, T_pose_source)
    pose_target = pose_from_matrix(T_pose_target, frame_id=frame_id)
    return pose_target

def pose_stamped2list(msg):
    return [float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
            float(msg.pose.orientation.x),
            float(msg.pose.orientation.y),
            float(msg.pose.orientation.z),
            float(msg.pose.orientation.w),
            ]

def matrix_from_pose(pose):
    pose_list = pose_stamped2list(pose)
    trans = pose_list[0:3]
    quat = pose_list[3:7]

    T = np.zeros((4, 4))
    T[-1, -1] = 1
    r = R.from_quat(quat)
    T[:3, :3] = r.as_dcm()
    T[0:3, 3] = trans
    return T

def farthest_point_downsample(pointcloud, k):
    '''                                                                                                                           
    pointcloud (numpy array): cartesian points, shape (N, 3)                                                                      
    k (int): number of points to sample                                                                                           
                                                                                                                                  
    sampled_cloud (numpy array): downsampled points, shape (k, 3)                                                                 
    '''
    start_ind = np.random.randint(0, len(pointcloud)) # pick a random point in the cloud to start                                 
    sampled_cloud = np.array([pointcloud[start_ind]])
    pointcloud = np.delete(pointcloud, start_ind, axis=0)
    mindists = np.full(len(pointcloud), np.inf) # make a list of minimum distances to samples for each point                      
    for i in range(k):
        last_sample = sampled_cloud[-1]
        ptdists = ((pointcloud-last_sample)**2).sum(axis=1) # distances between each point and most recent sample
        mindists = np.minimum(ptdists, mindists)
        min_ind = np.argmax(mindists)
        sampled_cloud = np.append(sampled_cloud, [pointcloud[min_ind]], axis=0)
        pointcloud = np.delete(pointcloud, min_ind, axis=0)
        mindists = np.delete(mindists, min_ind, axis=0)
    return sampled_cloud
