#!/usr/bin/python3

import numpy as np
import yarp
import sys
import pickle as pk
import distutils.util
import cv2

from functions.config import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_JOINTS
from functions.utilities import read_openpose_data, get_features
from functions.utilities import draw_on_img, create_bottle, get_mean_depth_over_area


yarp.Network.init()


class Classifier(yarp.RFModule):

    def configure(self, rf):
        self.model_name = rf.find("model_name").asString()
        print('SVM model file: %s' % self.model_name)
        self.clf = pk.load(open('./src/functions/' + self.model_name, 'rb'))
        self.MAX_FRAMERATE = bool(distutils.util.strtobool((rf.find("max_framerate").asString())))
        print('Max framerate: %s' % str(self.MAX_FRAMERATE))
        self.threshold = rf.find("max_propagation").asInt32()  # to reset the buffer
        print('SVM Buffer threshold: %d' % self.threshold)
        self.buffer = ('', (0, 0), 0, 0, 0)  # centroid, prediction and level of confidence
        self.counter = 0  # counter for the threshold
        self.svm_buffer_size = 3
        self.svm_buffer = []
        self.id_image = '%08d' % 0

        self.cmd_port = yarp.Port()
        self.cmd_port.open('/classifier/command:i')
        print('{:s} opened'.format('/classifier/command:i'))
        self.attach(self.cmd_port)

        # input port for rgb image
        self.in_port_human_image = yarp.BufferedPortImageRgb()
        self.in_port_human_image.open('/classifier/image:i')
        self.in_buf_human_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.in_buf_human_image = yarp.ImageRgb()
        self.in_buf_human_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.in_buf_human_image.setExternal(self.in_buf_human_array.data, self.in_buf_human_array.shape[1], self.in_buf_human_array.shape[0])
        print('{:s} opened'.format('/classifier/image:i'))

        # input port for depth
        self.in_port_human_depth = yarp.BufferedPortImageFloat()
        self.in_port_human_depth_name = '/classifier/depth:i'
        self.in_port_human_depth.open(self.in_port_human_depth_name)
        self.in_buf_human_depth_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.float32)
        self.in_buf_human_depth = yarp.ImageFloat()
        self.in_buf_human_depth.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.in_buf_human_depth.setExternal(self.in_buf_human_depth_array.data, self.in_buf_human_depth_array.shape[1], self.in_buf_human_depth_array.shape[0])
        print('{:s} opened'.format('/classifier/depth:i'))

        # input port for openpose data
        self.in_port_human_data = yarp.BufferedPortBottle()
        self.in_port_human_data.open('/classifier/data:i')
        print('{:s} opened'.format('/classifier/data:i'))

        # output port for the prediction
        self.out_port_prediction = yarp.Port()
        self.out_port_prediction.open('/classifier/pred:o')
        print('{:s} opened'.format('/classifier/pred:o'))

        # output port for rgb image
        self.out_port_human_image = yarp.Port()
        self.out_port_human_image.open('/classifier/image:o')
        self.out_buf_human_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.out_buf_human_image = yarp.ImageRgb()
        self.out_buf_human_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.out_buf_human_image.setExternal(self.out_buf_human_array.data, self.out_buf_human_array.shape[1], self.out_buf_human_array.shape[0])
        print('{:s} opened'.format('/classifier/image:o'))

        # output port for dumper
        self.out_port_human_image_dump = yarp.Port()
        self.out_port_human_image_dump.open('/classifier/dump:o')
        self.out_buf_human_array_dump = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.out_buf_human_image_dump = yarp.ImageRgb()
        self.out_buf_human_image_dump.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.out_buf_human_image_dump.setExternal(self.out_buf_human_array_dump.data, self.out_buf_human_array_dump.shape[1], self.out_buf_human_array_dump.shape[0])
        print('{:s} opened'.format('/classifier/dump:o'))

        self.human_image = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.human_image_depth = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.float32)

        return True

    def respond(self, command, reply):
        if command.get(0).asString() == 'quit':
            print('received command QUIT')
            self.cleanup()
            reply.addString('QUIT command sent')
        elif command.get(0).asString() == 'get':
            print('received command GET')
            self.out_buf_human_array_dump[:, :] = self.human_image
            self.out_port_human_image_dump.write(self.out_buf_human_image_dump)
            reply.copy(create_bottle(self.buffer))
        else:
            print('Command {:s} not recognized'.format(command.get(0).asString()))
            reply.addString('Command {:s} not recognized'.format(command.get(0).asString()))

        return True

    def cleanup(self):
        print('Cleanup function')
        self.in_port_human_image.close()
        self.in_port_human_data.close()
        self.in_port_human_depth.close()
        self.out_port_human_image.close()
        self.out_port_prediction.close()
        return True

    def interruptModule(self):
        print('Interrupt function')
        self.in_port_human_image.close()
        self.in_port_human_data.close()
        self.in_port_human_depth.close()
        self.out_port_human_image.close()
        self.out_port_prediction.close()
        return True

    def getPeriod(self):
        return 0.001

    def updateModule(self):

        received_image = self.in_port_human_image.read()
        received_depth = self.in_port_human_depth.read(False) # non blocking

        if received_image:
            self.in_buf_human_image.copy(received_image)
            human_image = np.copy(self.in_buf_human_array)

            self.human_image = np.copy(human_image)
            self.id_image = '%08d' % ((int(self.id_image) + 1) % 100000)

            if received_depth:
                self.in_buf_human_depth.copy(received_depth)
                self.human_image_depth = np.copy(self.in_buf_human_depth_array)

            if self.MAX_FRAMERATE:
                received_data = self.in_port_human_data.read(False)  # non blocking
            else:
                received_data = self.in_port_human_data.read()

            if received_data:
                poses, conf_poses, faces, conf_faces = read_openpose_data(received_data)
                # get features of all people in the image
                data = get_features(poses, conf_poses, faces, conf_faces)

                if data:
                    # predict model
                    # start from 2 because there is the centroid value in the position [0,1]
                    ld = np.array(data)
                    x = ld[:, 2:(NUM_JOINTS * 2) + 2]
                    c = ld[:, (NUM_JOINTS * 2) + 2:ld.shape[1]]
                    # weight the coordinates for its confidence value
                    wx = np.concatenate((np.multiply(x[:, ::2], c), np.multiply(x[:, 1::2], c)), axis=1)

                    # return a prob value between 0,1 for each class
                    y_classes = self.clf.predict_proba(wx)
                    # take only the person with id 0, we suppose that there is only one person in the scene
                    itP = 0
                    prob = max(y_classes[itP])
                    y_pred = (np.where(y_classes[itP] == prob))[0]

                    if len(self.svm_buffer) == self.svm_buffer_size:
                        self.svm_buffer.pop(0)

                    self.svm_buffer.append([y_pred[0], prob])

                    count_class_0 = [self.svm_buffer[i][0] for i in range(0, len(self.svm_buffer))].count(0)
                    count_class_1 = [self.svm_buffer[i][0] for i in range(0, len(self.svm_buffer))].count(1)
                    if (count_class_1 == count_class_0):
                        y_winner = y_pred[0]
                        prob_mean = prob
                    else:
                        y_winner = np.argmax([count_class_0, count_class_1])
                        prob_values = np.array(
                            [self.svm_buffer[i][1] for i in range(0, len(self.svm_buffer)) if self.svm_buffer[i][0] == y_winner])
                        prob_mean = np.mean(prob_values)

                    if self.human_image_depth is not None:
                        depth = get_mean_depth_over_area(self.human_image_depth, [int(ld[itP,0]), int(ld[itP,1])], 30)
                    else:
                        depth = -1

                    pred = create_bottle((self.id_image, (int(ld[itP,0]), int(ld[itP,1])), depth, y_winner, prob_mean))
                    human_image = draw_on_img(human_image, self.id_image, (ld[itP,0], ld[itP,1]), y_winner, prob_mean)

                    self.out_buf_human_array[:, :] = human_image
                    self.out_port_human_image.write(self.out_buf_human_image)
                    self.out_port_prediction.write(pred)

                    self.buffer = (self.id_image, (int(ld[itP,0]), int(ld[itP,1])), depth, y_winner, prob_mean)
                    self.counter = 0
                else:
                    pred = create_bottle((self.id_image, (), -1, -1, -1))
                    human_image = cv2.putText(human_image, 'id: ' + str(self.id_image), tuple([25, 30]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

                    # send in output only the image with prediction set to -1 (invalid value)
                    self.out_buf_human_array[:, :] = human_image
                    self.out_port_human_image.write(self.out_buf_human_image)
                    self.out_port_prediction.write(pred)

                    self.buffer = (self.id_image, (), -1, -1, -1)
                    self.counter = 0
            else:
                if self.MAX_FRAMERATE and (self.counter < self.threshold):
                    # send in output the buffer
                    self.buffer = (self.id_image, self.buffer[1], self.buffer[2], self.buffer[3], self.buffer[4])
                    if self.buffer[1]:
                        human_image = draw_on_img(human_image, self.buffer[0], self.buffer[1], self.buffer[3], self.buffer[4])
                    else:
                        human_image = cv2.putText(human_image, 'id: ' + str(self.id_image), tuple([25, 30]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

                    pred = create_bottle(self.buffer)

                    # write rgb image
                    self.out_buf_human_array[:, :] = human_image
                    self.out_port_human_image.write(self.out_buf_human_image)
                    self.out_port_prediction.write(pred)

                    self.counter = self.counter + 1

        return True


if __name__ == '__main__':

    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext("Classifier")
    rf.setDefaultConfigFile('./app/config/classifier_conf.ini')

    rf.configure(sys.argv)

    # Run module
    manager = Classifier()
    manager.runModule(rf)
