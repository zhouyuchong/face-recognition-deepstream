from enum import Enum, unique

@unique
class CounterOp(Enum):
    UP = "up"
    DOWN = "down"

@unique
class FaceState(Enum):
    Init = 0
    State1 = 1
    State2 = 2
    State3 = 3
    State4 = 4

@unique
class CarState(Enum):
    Init = 0
    READY = 1
    FINISHED = 2
    TERMINATE = 3

@unique
class ParkState(Enum):
    Init = 1
    MOVING = 2
    STILL = 3
    DONE = 4


@unique
class AnalyticsValue(Enum):
    LINECROSSING = 1
    ROIREGION = 2

@unique
class CamStatus(Enum):
    HOME = 1
    ZOOM = 2

PGIE = 1
SGIE = 2
MUXER_OUTPUT_WIDTH=1280
MUXER_OUTPUT_HEIGHT=720
# analytics values
ANALYTICS_CONFIG_FILE = '/opt/nvidia/deepstream/deepstream-6.1/samples/configs/deepstream-app/config_nvanalytics.txt'
# kafka lib
KAFKA_LIB = "/opt/nvidia/deepstream/deepstream-6.1/lib/libnvds_kafka_proto.so"
BOOTSTRAP_SERVER = 'localhost:9092'

# 检测的物体
DETECT_OBJECTS = [2]


