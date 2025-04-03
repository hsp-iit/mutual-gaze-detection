#!/usr/bin/python3
import os

root_dir = os.getcwd()
images_dir = '/images/rgb'
out_json_dir = '/images/openpose_json'
out_rend_dir = '/images/openpose_rgb'
# enter path where OpenPose was installed
os.chdir('/home/openpose')
# path for openpose deployment command
openpose_path = './build/examples/openpose/openpose.bin'

save_openpose_images = False

if not os.path.exists(out_json_dir):
    os.makedirs(out_json_dir)

    if save_openpose_images:
        if not os.path.exists(out_rend_dir):
            os.makedirs(out_rend_dir)

    if save_openpose_images:
        cmd = "%s --image_dir %s --display 0 --write_images %s --write_json %s --face --num_gpu 1 --num_gpu_start 1 --scale_number 4 --scale_gap 0.25" \
            % (openpose_path, images_dir, out_rend_dir, out_json_dir)
    else:
        cmd = "%s --image_dir %s --display 0 --render_pose 0 --write_json %s --face --num_gpu 1 --num_gpu_start 1 --scale_number 4 --scale_gap 0.25" \
              % (openpose_path, images_dir, out_json_dir)

        os.system(cmd)
else:
    print('OpenPose already run in the folder %s' % images_dir)