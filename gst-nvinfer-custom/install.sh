###
 # @Author: zhouyuchong
 # @Date: 2024-05-21 16:47:58
 # @Description: 
 # @LastEditors: zhouyuchong
 # @LastEditTime: 2024-08-19 15:49:17
### 

log() {
    local message="$1"
    current_time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[${current_time}] $message"
}

CUR_DIR=$PWD

log "[INFO]check current directory"
if [ ! -d "$CUR_DIR/gst-nvinfer" ]; then
  log "[ERROR]wrong execute directory!"
  exit 1
fi

log "[INFO]backing up original files"
if [ -d "$CUR_DIR/backup" ]; then
  log "[INFO]dir already exists"
else
  log "[INFO]making backup dir"
  mkdir backup
fi

if [ ! -f "$CUR_DIR/backup/libnvds_infer.so" ]; then
    log "[INFO]backup libnvds_infer.so"
    mv /opt/nvidia/deepstream/deepstream/lib/libnvds_infer.so ./backup/
else
    log "[INFO]libnvds_infer.so already backup"
fi

if [ ! -f "$CUR_DIR/backup/libnvdsgst_infer.so" ]; then
    log "[INFO]backup libnvdsgst_infer.so"
    mv /opt/nvidia/deepstream/deepstream/lib/gst-plugins/libnvdsgst_infer.so ./backup/libnvdsgst_infer.so
else
    log "[INFO]libnvdsgst_infer.so already backup"
fi

if [ ! -f "$CUR_DIR/backup/nvdsinfer.h" ]; then
    log "[INFO]backup nvdsinfer.h"
    mv /opt/nvidia/deepstream/deepstream/sources/includes/nvdsinfer.h ./backup/nvdsinfer.h
else
    log "[INFO]nvdsinfer.h already backup"
fi

cuda_version=$(nvcc --version | grep -oP 'V\K\d+\.\d+')
log "[INFO]CUDA Version: $cuda_version"

export CUDA_VER=$cuda_version

echo -n "Please input deepstream version(e.g. 6.0):"
read DEEPSTREAM_VERSION
export NVDS_VERSION=$DEEPSTREAM_VERSION

log '[INFO]Installing dependencies'
cp ./nvdsinfer.h /opt/nvidia/deepstream/deepstream/sources/includes/nvdsinfer.h

log '[INFO]Building nvdsinfer'
cd nvdsinfer
make -j$(nproc)
make install

log '[INFO]Building gst-nvinfer'
cd $CUR_DIR
cd gst-nvinfer
make -j$(nproc)
make install

log "[SUCCESS]install success"
