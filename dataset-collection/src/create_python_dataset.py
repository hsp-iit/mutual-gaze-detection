#!/usr/bin/python3

import os
import pandas as pd
from openpose_utilities import compute_head_face_features, read_openpose_from_json

# camera = "realsense"
camera = "icubCamera"


def create_python_dataset(type_exp):

    annotations_path = os.path.join(dataset, '%s_annotations.txt' % type_exp)
    output_file = os.path.join(dataset, 'feats_dataset_%s.pkl' % type_exp)

    annotations_file = open(annotations_path, "r")
    annotations_contents = annotations_file.readlines()
    annotations_contents = [x.strip() for x in annotations_contents]
    annotations = [x.split(" ") for x in annotations_contents]

    df = pd.DataFrame(columns=['participant', 'face_points', 'eye_contact'])

    # for each line in the annotation file
    for it in range(0, len(annotations)):
        # openpose features
        annotation = annotations[it]
        annotation_split = annotation[0].split("/")
        participant = annotation_split[1]
        sample = (annotation_split[-1]).replace('.jpg', '')

        folder_participant = os.path.join(dataset, participant)
        openpose_data_dir = os.path.join(folder_participant, '%s_data_openpose' % type_exp)

        openpose_file = openpose_data_dir + '/' + sample + '_keypoints.json'

        if os.path.exists(openpose_file):
            poses, conf_poses, faces, conf_faces = read_openpose_from_json(openpose_file)

            if poses is not None and poses != []:
                features, _ = compute_head_face_features(poses[0], conf_poses[0], faces[0], conf_faces[0])

                if features is not None:
                    eyecontact = annotation[1]
                    df = df.append({'participant': participant,
                                    'face_points': features,
                                    'eye_contact': eyecontact}, ignore_index=True)

                else:
                    print("None features: " + openpose_file)
                if it % 100 == 0:
                    df.to_pickle(output_file)
                    print('\n### Done with: %d \n' % it)
            else:
                print("None poses: " + openpose_file)
        else:
            print("No JSON: " + openpose_file)

    df.to_pickle(output_file)
    print('Dataframe saved.')


# ---------------------------------------------------------------

root_dir = os.path.join(os.getcwd(), "..", "datasets-preprocessing")
dataset = os.path.join(root_dir, camera)

create_python_dataset('eyecontact')
