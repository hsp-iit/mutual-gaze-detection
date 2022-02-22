#!/usr/bin/python3

import os

#camera = 'realsense'
camera = 'icubCamera'

thisPath = os.getcwd()
# enter the path where the images were saved
in_root_dir = os.path.join(thisPath, '..', 'datasets-preprocessing', camera)
# enter the path for the output
out_root_dir = os.path.join(thisPath, '..', 'datasets-preprocessing', camera)

# enter path where OpenPose was installed
os.chdir('/home/openpose')
# path for openpose deployment command
openpose_path = './build/examples/openpose/openpose.bin'

list_participant = [name for name in os.listdir(in_root_dir) if os.path.isdir(os.path.join(in_root_dir, name))]
list_participant.sort()
for participant in list_participant:
    print("Processing participant:", participant)
    in_folder_participant = os.path.join(in_root_dir, participant)
    out_folder_participant = os.path.join(out_root_dir, participant)

    out_json_dir = os.path.join(out_folder_participant, 'eyecontact_data_openpose')
    out_rend_dir = os.path.join(out_folder_participant, 'eyecontact_images_openpose')

    if not os.path.exists(out_json_dir):
        os.makedirs(out_json_dir)
    if not os.path.exists(out_rend_dir):
        os.makedirs(out_rend_dir)

    print("Processing eyecontact images")
    images_dir = os.path.join(in_folder_participant, 'eyecontact_images_human/eyecontact_condition')
    cmd = "%s --image_dir %s --display 0 --write_images %s --write_json %s --face --num_gpu 1 --num_gpu_start 1 --scale_number 4 --scale_gap 0.25" \
        % (openpose_path, images_dir, out_rend_dir, out_json_dir)

    os.system(cmd)

    print("Processing no eyecontact images")
    images_dir = os.path.join(in_folder_participant, 'eyecontact_images_human/no_eyecontact_condition')
    cmd = "%s --image_dir %s --display 0 --write_images %s --write_json %s --face --num_gpu 1 --num_gpu_start 1 --scale_number 4 --scale_gap 0.25" \
          % (openpose_path, images_dir, out_rend_dir, out_json_dir)

    os.system(cmd)

os.chdir(thisPath)

