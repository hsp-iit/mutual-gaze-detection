#!/usr/bin/python3
import os
import cv2
import numpy as np
import pickle as pk

from config import NUM_JOINTS
from utilities import read_openpose_from_json, get_features

root_dir = os.getcwd()
images_dir = './images/rgb'
openpose_json_dir = './images/openpose_json'
output_prediction = './images/output'
img_ext = '.jpg' # extension of the files
# load model
mutualgaze_classifier = pk.load(open('./model_svm_realsense.pkl', 'rb'))

if not os.path.exists(output_prediction):
    os.makedirs(output_prediction)

imgs = list(filter(lambda x: img_ext in x, os.listdir(images_dir)))
imgs = [img.replace(img_ext, '') for img in imgs]
imgs.sort()

for img in imgs:
    img_file = os.path.join(images_dir, img + img_ext)
    json_file = os.path.join(openpose_json_dir, img + '_keypoints.json')
    poses, conf_poses, faces, conf_faces = read_openpose_from_json(json_file)
    img_sample = cv2.imread(img_file, cv2.COLOR_BGR2RGB)
    if poses:
        data = get_features(poses, conf_poses, faces, conf_faces)
        if data:
            ld = np.array(data)
            x = ld[:, 2:(NUM_JOINTS * 2) + 2]
            c = ld[:, (NUM_JOINTS * 2) + 2:ld.shape[1]]
            wx = np.concatenate((np.multiply(x[:, ::2], c), np.multiply(x[:, 1::2], c)), axis=1)
            y_classes = mutualgaze_classifier.predict_proba(wx)
            itP = 0
            prob = max(y_classes[itP])
            y_pred = (np.where(y_classes[itP] == prob))[0]

            txt = 'eye contact %s, c %0.2f' % (('YES' if y_pred == 1 else 'NO'), prob)
            img_sample = cv2.putText(img_sample, txt, tuple([20, 40]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(output_prediction, img + '.png'), img_sample)
    else:
        print('No human detected for %s' % img_file)

