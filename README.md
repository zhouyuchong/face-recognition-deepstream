<!--
 * @Author: zhouyuchong
 * @Date: 2024-08-19 14:13:02
 * @Description: 
 * @LastEditors: zhouyuchong
 * @LastEditTime: 2024-09-19 14:27:00
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
+ Opencv

Follow [deepstream](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html#dgpu-setup-for-ubuntu) official doc to install dependencies.

Deepstream docker is more recommended.

## Models
+ [yolov8-face](https://github.com/derronqi/yolov8-face)
+ [retinaface](https://github.com/biubug6/Pytorch_Retinaface)
+ [arcface](https://github.com/deepinsight/insightface/releases/tag/v0.7)

## Alignment

[gst-nvinfer-custom](https://github.com/zhouyuchong/gst-nvinfer-custom)

## Usage
### 1 - prepare data
+ put the feature file(.npy format) to `data/known_faces`
+ or put face images to `data/unknown_faces` and run `python3 utils/gen_feature.py`
### 2 - compile 
+ `gst-nvinfer-custom` : follow [README](https://github.com/zhouyuchong/gst-nvinfer-custom)
+ `nvdsinfer_customparser` for detector post-process
### 3 - run 
```
python3 main.py
```

## References
+ [NVIDIA-AI-IOT/deepstream_python_apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps)
+ [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)
+ [yolov8-face](https://github.com/derronqi/yolov8-face)
+ [arcface](https://github.com/deepinsight/insightface)

