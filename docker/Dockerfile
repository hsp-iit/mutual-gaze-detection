# Use the base image you specified
ARG DOCKER_SRC=nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
FROM ${DOCKER_SRC}

ENV DEBIAN_FRONTEND="noninteractive"

# Install necessary packages
USER root
RUN apt-get update && apt-get install -y sudo pulseaudio libasound2-plugins && apt-get clean

RUN apt-get update
RUN apt-get install -y python3-testresources

RUN apt-get install -y libgoogle-glog-dev libboost-all-dev libhdf5-serial-dev libatlas-base-dev
RUN apt-get install -y libcanberra-gtk-module libcanberra-gtk3-module
RUN apt-get install -y protobuf-compiler libprotobuf-dev

RUN mkdir -p /etc/bash_completion.d/ && apt-get update && apt-get install -y \
       # MISC
       bash-completion qt5-default git cmake cmake-curses-gui build-essential vim terminator
RUN apt-get install -y \
        # YARP
        libace-dev libeigen3-dev libgsl0-dev libedit-dev libqcustomplot-dev qtmultimedia5-dev qtdeclarative5-dev \
        libqt5opengl5-dev qttools5-dev libtinyxml-dev
RUN apt-get install -y \
        #yarp-gui
        qtbase5-dev qtdeclarative5-dev qtmultimedia5-dev \
        qml-module-qtquick2 qml-module-qtquick-window2 qml-module-qtmultimedia qml-module-qtquick-dialogs \
        qml-module-qtquick-controls qml-module-qt-labs-folderlistmodel qml-module-qt-labs-settings
        #core
RUN apt-get install -y libboost-system-dev libboost-filesystem-dev libboost-thread-dev
RUN apt-get install -y liblua5.1-dev lua5.1 libsdl1.2-dev libxml2-dev
RUN apt-get install -y coinor-libipopt-dev libode-dev libopencv-dev


#### Latest version Cmake ####
RUN apt-get install -y apt-transport-https ca-certificates gnupg software-properties-common wget &&\
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add - &&\
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' &&\
    apt-get update &&\
    apt-get install -y kitware-archive-keyring &&\
    apt-key --keyring /etc/apt/trusted.gpg del C1F34CDD40CD72DA &&\
    apt-get install -y cmake \
    &&\
    rm -rf /var/lib/apt/lists/*


ARG PROJECTS_DIR=/home/code

RUN mkdir /home
RUN mkdir ${PROJECTS_DIR}

ARG SWIG_TAG=v4.0.2

#### Swig 4 ####
RUN apt-get update && apt-get install -y automake byacc bison
RUN cd ${PROJECTS_DIR} &&\
    git clone https://github.com/swig/swig.git &&\
    cd swig &&\
    git checkout tags/${SWIG_TAG}
RUN cd ${PROJECTS_DIR}/swig &&\
    ./autogen.sh &&\
    ./configure &&\
    make && make install


ARG ROBOTOLOGY_SUPERBUILD_TAG=v2022.05.0 

#### Build robotology superbuild ####
RUN cd ${PROJECTS_DIR} &&\
    git clone https://github.com/robotology/robotology-superbuild.git &&\
    cd robotology-superbuild &&\
    git checkout tags/${ROBOTOLOGY_SUPERBUILD_TAG}
RUN cd ${PROJECTS_DIR}/robotology-superbuild &&\
    mkdir build && cd build &&\
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DNON_INTERACTIVE_BUILD:BOOL=ON \
        -DROBOTOLOGY_ENABLE_CORE:BOOL=ON \
        -DROBOTOLOGY_USES_PYTHON=ON \
        -DROBOTOLOGY_USES_GAZEBO=OFF \
        -DYCM_EP_ADDITIONAL_CMAKE_ARGS:STRING="-DICUB_COMPILE_BINDINGS:BOOL=ON -DCREATE_PYTHON:BOOL=ON -DENABLE_yarpmod_rgbdSensor_nws_yarp:BOOL=ON -DENABLE_yarpmod_RGBDSensorClient:BOOL=ON -DENABLE_yarpcar_mjpeg:BOOL=ON -DENABLE_yarppm_depthimage_to_rgb:BOOL=ON -DENABLE_yarppm_depthimage_compression_zlib:BOOL=ON" ../ && \
        make -j4

ENV PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3/dist-packages
ENV PYTHONPATH=$PYTHONPATH:${PROJECTS_DIR}/robotology-superbuild/build/src/YARP/lib/python3

ENV PATH=$PATH:${PROJECTS_DIR}/robotology-superbuild/build/install/bin
ENV YARP_DATA_DIRS=/home/code/robotology-superbuild/build/install/share/yarp/

##### Realsense SDK #####
RUN apt-get update
RUN apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
RUN add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u
RUN apt-get update
RUN apt-get install -y librealsense2-dkms librealsense2-utils
RUN apt-get install -y librealsense2-dev librealsense2-dbg
RUN apt-get install -y libv4l-dev


##### yarp device realsense #####
RUN cd ${PROJECTS_DIR} &&\
    git clone https://github.com/robotology/yarp-device-realsense2
RUN cd ${PROJECTS_DIR}/yarp-device-realsense2 &&\
    mkdir build &&\
    cd ./build &&\
    cmake .. \
        -DCMAKE_INSTALL_PREFIX=${PROJECTS_DIR}/robotology-superbuild/build/install &&\
    make -j${nproc} && make install


##### Openpose #####
ARG OPENPOSE_TAG=v1.6.0

RUN cd ${PROJECTS_DIR} &&\
    git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose &&\
    cd ./openpose/ &&\
    git submodule update --init --recursive --remote &&\
    git checkout tags/${OPENPOSE_TAG}
COPY openpose-models/ ${PROJECTS_DIR}/openpose/models/
RUN cd ${PROJECTS_DIR}/openpose &&\
    mkdir build &&\
    cd ./build &&\
    cmake .. -DDOWNLOAD_HAND_MODEL=OFF &&\
    make -j$(nproc) && make install


##### YarpOpenPose #####
RUN cd ${PROJECTS_DIR} &&\
    git clone https://github.com/robotology/human-sensing
RUN cd ${PROJECTS_DIR}/human-sensing/yarpOpenPose &&\
    mkdir build &&\
    cd ./build &&\
    cmake .. &&\
    make && make install


##### Python dependencies ####
RUN apt-get update
RUN apt-get install -y python3-pip
RUN python3.6 -m pip install --upgrade pip
RUN python3.6 -m pip install --upgrade setuptools
RUN python3.6 -m pip install --upgrade wheel

RUN python3.6 -m pip install numpy opencv-contrib-python yarp pandas
RUN python3.6 -m pip install scikit-learn==0.24.1


##### Mutual gaze #####
RUN cd ${PROJECTS_DIR} &&\
    git clone hhttps://github.com/hsp-iit/mutual-gaze-detection
RUN cd ${PROJECTS_DIR}/mutual-gaze-detection/classifier-demo &&\
    mkdir build &&\
    cd ./build &&\
    cmake .. &&\
    make && make install

