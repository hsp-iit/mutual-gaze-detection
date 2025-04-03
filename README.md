# Mutual Gaze Detection
This repository contains the collected datasets, the Python code needed to collect them,
the learning trained models of the mutual gaze classifier and the Python code to run it. 
The whole repository is related to the paper 
_Toward an attentive robotic architecture: learning-based mutual gaze estimation 
in human-robot interaction_.

## Abstract
Social robotics is an emerging field that is expected to grow rapidly in the near future. In
fact, it is increasingly more frequent to have robots that operate in close proximity with
humans or even collaborate with them in joint tasks. In this context, the investigation of how
to endow a humanoid robot with social behavioral skills typical of human–human
interactions is still an open problem. Among the countless social cues needed to
establish a natural social attunement, this article reports our research toward the
implementation of a mechanism for estimating the gaze direction, focusing in particular
on mutual gaze as a fundamental social cue in face-to-face interactions. We propose a
learning-based framework to automatically detect eye contact events in online interactions
with human partners. The proposed solution achieved high performance both in silico and
in experimental scenarios. Our work is expected to be the first step toward an attentive
architecture able to endorse scenarios in which the robots are perceived as social partners.

## Requirements
The following modules are required for the proposed pipeline.
#### OpenPose
This is the requirement to detect anatomical key-points. Please follow the instructions provided in the [OpenPose repository](https://github.com/CMU-Perceptual-Computing-Lab/openpose) for proper installation. To run Openpose you may need a GPU. In this work we used an NVIDIA GTX 1080Ti. 

## How to run it
The folder _classifier-demo-offline_ contains a sample demo code in python to show how to use the classifier.
- Create a python virtual environment running <code>python3 -m venv myenv</code>
- Launch the virtual env with source <code>myenv/bin/activate</code> and then <code>pip3 install -r requirements.txt</code>
- Run first <code>python3 run_openpose.py</code> to preprocess the images in the folder the folder _./images/rgb_ and extract the facial keypoint from them in json files. They will be saved in the folder _./images/openpose_json_
- Run <code>python3 demo.py</code>, the resulting images will be saved in the folder _./images/output_

The folder _classifier-demo_ contains the code to use the classifier with the middleware Yarp and with a robotic platform. For that you need to have also Yarp installed on the machine following the instructions at <url>https://www.yarp.it/latest/yarp_installation.html</url>. Alternatively, we suggest to use the dockerfile in the folder _docker_ where there are all the dependencies and the modules need for a right setup.

## Reference
If you use any data or code in this repository, please, cite as following:

```
@article{lombardi2022,
  title={Toward an attentive robotic architecture: learning-based mutual gaze estimation in human-robot interaction},
  author={Lombardi, Maria and Maiettini, Elisa and De Tommaso, Davide and Wykowska, Agnieszka and Natale, Lorenzo},
  journal={Frontiers in Robotics and AI},
  publisher={Frontiers}
}
```