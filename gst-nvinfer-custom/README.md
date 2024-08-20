<!--
 * @Author: zhouyuchong
 * @Date: 2024-02-26 14:51:58
 * @Description: 
 * @LastEditors: zhouyuchong
 * @LastEditTime: 2024-05-23 14:34:51
-->
# Custom gst-nvinfer (DEMO)
This is a custom gst-nvinfer plugin to do some preprocess and postprocess.

## Requirements
+ Deepstream 6.0+
+ Opencv

## Notice
This demo supports models:
+ [Retinaface](https://github.com/wang-xinyu/tensorrtx/tree/master/retinaface)
+ [Retina_License_Plate](https://github.com/gm19900510/Pytorch_Retina_License_Plate)
+ [Hyperlpr](https://github.com/szad670401/HyperLPR)

If want to use other models, codes in `tensor_extractor.cpp` should be modified for extracting landmarks from original tensor-output and `align_funcitons`.


## Feature
for detector
1. add landmarks in `nvdsinfer` so it can be processed together with bondingboxes in `nvdsinfer_customparser`
2. if `numLmks` not 0, we concat landmarks to the end of *labels*
3. decode landmarks in `attach_metadata_detector` and attach them to user metadata

for classifier

3. decode landmarks in object_user_metadata
4. use opencv to do alignment and replace original surface in gpu memory
5. Done!

## Usage
**A backup is strongly recommended!!!!**
1. replace `nvdsinfer.h`. It's under `/opt/nvidia/deepstream/deepstream/sources/includes` in official docker.
2. set cuda environment
```
export CUDA_VER=11.6
```
3. modify nvdsinfer_parser
   set `oinfo.numLmks` and `oinfo.landmarks` properties
4. compile nvdsinfer
```
cd nvdsinfer
make
make install
```
NOTE: To compile the sources, run make with "sudo" or root permission.

5. compile gst-nvinfer
```
cd gst-nvinfer
make
make install
```

6. set config file

for all detectors, set `cluster-mode=4` since we only modify `fillUnclusteredOutput`.
```
cluster-mode=4
```
for detector which output landmarks
```
enable-output-landmark = 1
```
for classifier which generate feature
use kyewords
+ alignment-type: 
  + 1: face -> [Retinaface](https://github.com/wang-xinyu/tensorrtx/tree/master/retinaface)
  + 2: license plate -> [Retina_License_Plate](https://github.com/gm19900510/Pytorch_Retina_License_Plate)
+ alignment-pics-path:
  path to save pictures


Example
```
alignment-type=2
```

## Comparison
License Plate
![car](./images/car.png)
Saved input NvBufSurface

![plate](./images/plate.png)

## TODO
+ use [npp](https://docs.nvidia.com/cuda/npp/group__affine__transform.html#ga5e722e6c67349032d4cacda4a696c237) to do alignment
+ improve stucture of passing lmks