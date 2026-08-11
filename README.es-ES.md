

<!--
 * @Author: zhouyuchong
 * @Date: 2024-08-19 14:13:02
 * @Description: 
 * @LastEditors: zhouyuchong
 * @LastEditTime: 2024-09-19 14:27:00
-->
# Reconocimiento facial con Deepstream
Esta es una pipeline de demostración de detección y reconocimiento facial construida sobre Deepstream.

## Requisitos
+ Deepstream 6.0+
+ GStreamer 1.14.5+
+ Cuda 11.4+
+ NVIDIA driver 470.63.01+
+ TensorRT 8+
+ Python 3.6+
+ Opencv

Sigue la [documentación oficial de deepstream](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html#dgpu-setup-for-ubuntu) para instalar las dependencias.

Se recomienda el contenedor Docker de Deepstream.

## Modelos
+ [yolov8-face](https://github.com/derronqi/yolov8-face)
+ [retinaface](https://github.com/biubug6/Pytorch_Retinaface)
+ [arcface](https://github.com/deepinsight/insightface/releases/tag/v0.7)

## Alineación

[gst-nvinfer-custom](https://github.com/zhouyuchong/gst-nvinfer-custom)

## Uso
### 1 - Preparar datos
+ coloca el archivo de características (formato .npy) en `data/known_faces`
+ o coloca las imágenes faciales en `data/unknown_faces` y ejecuta `python3 utils/gen_feature.py`
### 2 - Compilar 
+ `gst-nvinfer-custom` : sigue el [README](https://github.com/zhouyuchong/gst-nvinfer-custom)
+ `nvdsinfer_customparser` para el postprocesamiento del detector
### 3 - Ejecurar 
```
python3 main.py
```

## Referencias
+ [NVIDIA-AI-IOT/deepstream_python_apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps)
+ [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)
+ [yolov8-face](https://github.com/derronqi/yolov8-face)
+ [arcface](https://github.com/deepinsight/insightface)
