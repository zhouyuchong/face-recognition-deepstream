###
 # @Author: zhouyuchong
 # @Date: 2024-05-23 14:37:07
 # @Description: 
 # @LastEditors: zhouyuchong
 # @LastEditTime: 2024-05-23 15:43:02
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

log "[CHECK]check backup dir"
if [ ! -d "$CUR_DIR/backup" ]; then
  log "[INFO]no backup dir"
  exit 1
fi

if [ ! -f "$CUR_DIR/backup/libnvds_infer.so" ]; then
    log "[ERROR]missing file libnvds_infer.so"
    exit 1
else
    log "[INFO]restore libnvds_infer.so"
    mv ./backup/libnvds_infer.so /opt/nvidia/deepstream/deepstream/lib/libnvds_infer.so 
fi

if [ ! -f "$CUR_DIR/backup/libnvdsgst_infer.so" ]; then
    log "[ERROR]missing file libnvdsgst_infer.so"
    exit 1
else
    log "[INFO]restore libnvdsgst_infer.so"
    mv ./backup/libnvdsgst_infer.so /opt/nvidia/deepstream/deepstream/lib/gst-plugins/libnvdsgst_infer.so
fi

if [ ! -f "$CUR_DIR/backup/nvdsinfer.h" ]; then
    log "[INFO]missing file nvdsinfer.h"
    exit 1
else
    log "[INFO]restore nvdsinfer.h"
    mv ./backup/nvdsinfer.h /opt/nvidia/deepstream/deepstream/sources/includes/nvdsinfer.h 
fi

rm -rf ./backup

log "[SUCCESS]restore success"