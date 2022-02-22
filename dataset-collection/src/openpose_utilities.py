#!/usr/bin/python3

import numpy as np
from config import JOINTS_POSE, JOINTS_FACE
import json


def compute_centroid(points):
    mean_x = np.mean([p[0] for p in points])
    mean_y = np.mean([p[1] for p in points])

    return [mean_x, mean_y]


def joint_set(p, c):
    return (p[0] != 0.0 or p[1] != 0.0) and c > 0.1


def dist_2d(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)

    squared_dist = np.sum((p1 - p2)**2, axis=0)
    return np.sqrt(squared_dist)


def compute_head_face_features(pose, conf_pose, face, conf_face):

    n_joints_set = [pose[joint] for joint in JOINTS_POSE if joint_set(pose[joint], conf_pose[joint])]
    n_joints_set.extend([face[joint] for joint in JOINTS_FACE if joint_set(face[joint], conf_face[joint])])

    if len(n_joints_set) < 2:
        return None, None

    centroid = compute_centroid(n_joints_set)
    max_dist = max([dist_2d(j, centroid) for j in n_joints_set])

    new_repr_pose = [(np.array(pose[joint]) - np.array(centroid)) for joint in JOINTS_POSE]
    new_repr_face = ([(np.array(face[joint]) - np.array(centroid)) for joint in JOINTS_FACE])

    result = []

    for i in range(0, len(JOINTS_POSE)):

        if joint_set(pose[JOINTS_POSE[i]], conf_pose[JOINTS_POSE[i]]):
            result.append([new_repr_pose[i][0] / max_dist, new_repr_pose[i][1] / max_dist])
        else:
            result.append([0, 0])

    for i in range(0, len(JOINTS_FACE)):
        if joint_set(face[JOINTS_FACE[i]], conf_face[JOINTS_FACE[i]]):
            result.append([new_repr_face[i][0] / max_dist, new_repr_face[i][1] / max_dist])
        else:
            result.append([0, 0])

    flat_list = [item for sublist in result for item in sublist]

    for j in JOINTS_POSE:
        if conf_pose[j] > 0.1:
            flat_list.append(conf_pose[j])
        else:
            flat_list.append(0)

    for j in JOINTS_FACE:
        if conf_face[j] > 0.1:
            flat_list.append(conf_face[j])
        else:
            flat_list.append(0)

    return flat_list, centroid


def read_openpose_from_json(json_filename):

    with open(json_filename) as data_file:
        loaded = json.load(data_file)

        poses = []
        conf_poses = []
        faces = []
        conf_faces = []

        for arr in loaded["people"]:
            conf_poses.append(arr["pose_keypoints_2d"][2::3])
            arr_poses = np.delete(arr["pose_keypoints_2d"], slice(2, None, 3))
            poses.append(list(zip(arr_poses[::2], arr_poses[1::2])))

            conf_faces.append(arr["face_keypoints_2d"][2::3])
            arr_faces = np.delete(arr["face_keypoints_2d"], slice(2, None, 3))  # remove confidence values from the array
            faces.append(list(zip(arr_faces[::2], arr_faces[1::2])))

    return poses, conf_poses, faces, conf_faces

