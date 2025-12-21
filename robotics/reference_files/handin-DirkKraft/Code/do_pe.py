import open3d as o3d
import copy
import numpy as np
from tqdm import tqdm
import random

def do_pose_estimation(scene_pointcloud, object_pointcloud):
    # Dirk's code was here
    return np.identity(4)