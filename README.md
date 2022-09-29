# Face recognition with Deepstream 6.1
This is a object detection and face recognition app build on Deepstream.

## Requirements
+ Deepstream 6.0+
+ GStreamer 1.14.5+
+ Cuda 11.4+
+ NVIDIA driver 470.63.01+
+ TensorRT 8+
+ Python 3.6+

Follow [deepstream](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html#dgpu-setup-for-ubuntu) official doc to install dependencies.

Deepstream docker is more recommended.

## Pretrained
Please refer to links below for pretrained models and serialized TensorRT engine. Or download from [Google driver](https://drive.google.com/drive/folders/1HTdIhGrKP7JnKY6n8F95mI7SBnx7-4R3).
+ ~~[yolov5](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5)~~
+ [retinaface](https://github.com/wang-xinyu/tensorrtx/tree/master/retinaface)
+ [arcface](https://github.com/wang-xinyu/tensorrtx/tree/master/arcface)

## Alignment
there should be a face alignment before arcface. Use a custom-gst-nvinfer to preprocess the tensor-meta of retinaface. 

[Demo custom-gst-nvinfer](https://github.com/zhouyuchong/gst-nvinfer-custom)

## Usage
### 1 - compile retinaface parse function
```
cd models/retinaface
make
```
### 2 - modify all path in config and codes
+ all path in src/config/face/
+ line 24 in src/kbds/app/face.py

### 3 - Start kafka service before everything. 
Then set "ip;port;topic" in codes.
e.g.
```
DSFace("localhost;9092;deepstream")
```
### 4 - start app
```
python3 test/face_test_demo.py
```
Notice that only support RTSP stream and Gstreamer format.

### 5 - check result
images before and after alignment will be stored in images.


modify codes in test/cal_sim.py. replace with your own face-feature npy file.
```
python3 test/cal_sim.py
```
it will outputs all score.


## References
+ [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx)
+ [NVIDIA-AI-IOT/deepstream_python_apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps)

