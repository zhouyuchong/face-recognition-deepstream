<!--
 * @Author: zhouyuchong
 * @Date: 2024-08-19 14:13:02
 * @Description: 
 * @LastEditors: zhouyuchong
 * @LastEditTime: 2024-08-20 13:40:58
-->
# Face recognition with Deepstream
This is a face detection and recognition demo pipeline build on Deepstream.

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
+ ~~[retinaface](https://github.com/wang-xinyu/tensorrtx/tree/master/retinaface)~~
+ [retinaface](https://github.com/biubug6/Pytorch_Retinaface)
+ [arcface](https://github.com/wang-xinyu/tensorrtx/tree/master/arcface)

## Alignment
there should be a face alignment before arcface. Use a `gst-nvinfer-custom` to preprocess the tensor-meta of retinaface. 

[Demo custom-gst-nvinfer](https://github.com/zhouyuchong/gst-nvinfer-custom)

## Usage
### 1 - put the feature file to data/known_faces
### 2 - compile `gst-nvinfer-custom` if do alignment, care that the post-process should also be modified if use `gst-nvinfer-custom`
### 3 - modify config file under `config` folder
### 4 - run `python3 main.py`

## References
+ [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx)
+ [NVIDIA-AI-IOT/deepstream_python_apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps)
+ [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)

