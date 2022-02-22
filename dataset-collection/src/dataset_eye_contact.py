#!/usr/bin/python3

import numpy as np
import yarp
from pynput.keyboard import Key, Listener
from config import IMAGE_WIDTH, IMAGE_HEIGHT, ICUB_OFF


def on_release(key):
    global next_id

    try:
        if key == Key.space:
            print(' Button pressed - Taking the picture...')

            # Sending output image to data dumper
            print('Sending output image from realsense to data dumper...')
            out_buf_realsense_array[:, :] = realsense_image
            out_port_realsense_image.write(out_buf_realsense_image)

            if not ICUB_OFF:
                print('Sending output image from icub camera to data dumper...')
                out_buf_eye_array[:, :] = eye_image
                out_port_eye_image.write(out_buf_eye_image)

            next_id = True
    except AttributeError:
        # do something when a certain key is pressed, using key, not key.char
        pass

###################################################################

# Initialise YARP Network
yarp.Network.init()
# Add listener to handle input from the keyboard
listener = Listener(on_release=on_release)
listener.start()
next_id = False

###################################################################

if __name__ == '__main__':

    # Input port for the image
    in_port_realsense_image = yarp.BufferedPortImageRgb()
    in_port_realsense_image_name = '/human/realsense/image:i'
    in_port_realsense_image.open(in_port_realsense_image_name)
    in_buf_realsense_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    in_buf_realsense_image = yarp.ImageRgb()
    in_buf_realsense_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
    in_buf_realsense_image.setExternal(in_buf_realsense_array.data, in_buf_realsense_array.shape[1],
                                       in_buf_realsense_array.shape[0])

    # Output port for data dumper (image)
    out_port_realsense_image = yarp.Port()
    out_port_realsense_image_name = '/human/realsense/image:o'
    out_port_realsense_image.open(out_port_realsense_image_name)
    out_buf_realsense_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    out_buf_realsense_image = yarp.ImageRgb()
    out_buf_realsense_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
    out_buf_realsense_image.setExternal(out_buf_realsense_array.data, out_buf_realsense_array.shape[1],
                                        out_buf_realsense_array.shape[0])

    if not ICUB_OFF:
        in_port_eye_image = yarp.BufferedPortImageRgb()
        in_port_eye_image_name = '/human/right_eye/image:i'
        in_port_eye_image.open(in_port_eye_image_name)
        in_buf_eye_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        in_buf_eye_image = yarp.ImageRgb()
        in_buf_eye_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        in_buf_eye_image.setExternal(in_buf_eye_array.data, in_buf_eye_array.shape[1], in_buf_eye_array.shape[0])

        out_port_eye_image = yarp.Port()
        out_port_eye_image_name = '/human/right_eye/image:o'
        out_port_eye_image.open(out_port_eye_image_name)
        out_buf_eye_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        out_buf_eye_image = yarp.ImageRgb()
        out_buf_eye_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        out_buf_eye_image.setExternal(out_buf_eye_array.data, out_buf_eye_array.shape[1], out_buf_eye_array.shape[0])

    try:
        print("Press bar space to take the picture when ready...")

        while True:
            # Receiving input image
            received_realsense_image = in_port_realsense_image.read()

            if not received_realsense_image:
                print('Received image not valid.')
                continue

            if not ICUB_OFF:
                received_eye_image = in_port_eye_image.read()
                if not received_eye_image:
                    continue

                in_buf_eye_image.copy(received_eye_image)
                eye_image = np.copy(in_buf_eye_array)

            in_buf_realsense_image.copy(received_realsense_image)
            realsense_image = np.copy(in_buf_realsense_array)

    except KeyboardInterrupt:  # if ctrl-C is pressed

        print('Closing ports...')
        in_port_realsense_image.close()
        out_port_realsense_image.close()
        if not ICUB_OFF:
            in_port_eye_image.close()
            out_port_eye_image.close()
        print('Stopping listener...')
        listener.stop()
