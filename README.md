# Face recognition with Deepstream 6.0
This is a object detection and face recognition app build on Deepstream.

## Requirements
+ Deepstream 6.0
+ GStreamer 1.14.5
+ Cuda 11.4+
+ NVIDIA driver 470.63.01+
+ TensorRT 8+
+ Python 3.6
+ kafka

Follow [deepstream](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html#dgpu-setup-for-ubuntu) official doc to install dependencies.

Deepstream docker is more recommended.

Download [kafka](https://kafka.apache.org/downloads) binary source and follow [quick start](https://kafka.apache.org/quickstart) to start kafka service.

## Pretrained
Please refer to links below for pretrained models and serialized TensorRT engine. Or download from [Google driver](https://drive.google.com/drive/folders/1HTdIhGrKP7JnKY6n8F95mI7SBnx7-4R3).
+ [yolov5](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5)
+ [retinaface](https://github.com/wang-xinyu/tensorrtx/tree/master/retinaface)
+ [arcface](https://github.com/wang-xinyu/tensorrtx/tree/master/arcface)



## Usage
1. STEP 1

Start kafka service.

```
bin/zookeeper-server-start.sh config/zookeeper.properties
```
In a new terminal.
```
bin/kafka-server-start.sh config/server.properties
```
Or run the shell script in kafka/.

2. STEP 2

To run the app
```
LD_PRELOAD=../../models/yolov5/yolov5s/libYoloV5Decoder.so:../../models/retinaface/libRetinafaceDecoder.so:../../models/arcface/libArcFaceDecoder.so python3 main.py {URIPATH}
```
e.g.
```
LD_PRELOAD=../../models/yolov5/yolov5s/libYoloV5Decoder.so:../../models/retinaface/libRetinafaceDecoder.so:../../models/arcface/libArcFaceDecoder.so python3 main.py file:///opt/nvidia/deepstream/deepstream-6.0/sources/pythonapps/videos/samplevideo.h264
```
Notice that only support RTSP stream and Gstreamer format.
## References
+ [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx)
+ [NVIDIA-AI-IOT/deepstream_python_apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps)

## Known issue
+ get a large amount of wrong outputs

That's because of the [sychronazation bug](https://github.com/wang-xinyu/tensorrtx/commit/e72d9db48ba8453fd4465048a0175621f1b1c501#diff-e4f7cf998c56a033573edc39c7736317f73a28402d835ee44001bac64f386dfb) of tensorrtx codes. To solve it you should modify the [decode.cu](https://github.com/wang-xinyu/tensorrtx/blob/master/retinaface/decode.cu) file in tensorrtx repo and regenerate the engine.

## Undone
+ 
