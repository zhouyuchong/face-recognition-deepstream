import init_path

from kbds.app import DSFace
from kbds import Stream

import faulthandler
# 在import之后直接添加以下启用代码即可
faulthandler.enable()
import traceback
import time

# test app
app = DSFace("localhost;9092;deepstream")
print("--------- start app")
print(app.start())
time.sleep(2)

# print("--------- stop app")
# print(app.del_src())
# time.sleep(2)

# print("--------- start app")
# print(app.start())
# time.sleep(2)

# id = 0
# while True:
#     if id == 0:
#         name = "hx#camera#001"
#         # src = Stream(name, "file:///media/LumphiniPark_HD_007.mp4")
#         src = Stream(name, "rtsp://admin:sh123456@192.168.1.237:554/h264/ch1/main/av_stream")
#         app.add_src(src)
#         time.sleep(1)conn_str
#         id = id + 1
#     elif id == 1:
#         name = "hx#camera#kkk"
#         # src = Stream(name, "file:///media/test.ts")
#         src = Stream(name, "rtsp://admin:sh123456@192.168.1.235:554/h264/ch1/main/av_stream")
#         app.add_src(src)
#         time.sleep(1)
#         # app.play_src(id)
#         id = id + 1
    
#     elif id == 2:
#         src = Stream(id, "rtsp://admin:sh123456@192.168.1.237:554/h264/ch1/main/av_stream")
#         app.add_src(src)
#         time.sleep(1)
#         # app.play_src(id)
#         id = id + 1

#     elif id == 3:
#         src = Stream(id, "rtsp://admin:sh123456@192.168.1.237:554/h264/ch1/main/av_stream")
#         # src = Stream(id, "file:///media/test.ts")
#         app.add_src(src)
#         time.sleep(1)
#         # app.play_src(id)
#         id = id + 1

#     if id == 4:
#         print("delete src 1 ==========================================")
#         app.del_src("hx#camera#kkk")

#         time.sleep(3)

#         print("delete src 2 ==========================================")
#         app.del_src(2)
        
#         time.sleep(3)

#         print("delete src 3 ==========================================")
#         app.del_src(3)
        
#         time.sleep(3)

#         print("delete src 0 ==========================================")
#         app.del_src("hx#camera#001")

#         time.sleep(5)

#         name = "hx#camera#test"
#         # src = Stream(name, "rtsp://admin:sh123456@192.168.1.237:554/h264/ch1/main/av_stream")
#         src = Stream(name, "file:///media/3market_44.mp4")
#         app.add_src(src)

#         id = id + 1

    
    
    # 
    # print("-------------------")
    # app.pause_src(id=id)
    # time.sleep(20)
    # print("-------------------")
    
    # time.sleep(5)
    # if id == 3:
    #     break
    # id = id + 1

# while id >= 1:
#     app.del_src(id)
#     time.sleep(3)
#     id = id -1

print("-----------------------------------------------------------------------------")
id = 0
while True:
    if id == 0:
        str_id = "task-{}".format(id)
        # src = Stream(str_id, "file:///media/3market_33.mp4")
        
        # src = Stream(str_id, "file:///media/output.mp4")
        # src = Stream(str_id, "file:///media/ke.mp4")
        # src = Stream(str_id, "file:///media/pedestrian_London_Streets.mp4")
        # src = Stream(str_id, "rtsp://admin:sh123456@192.168.1.234:554/h264/ch1/main/av_stream")
        src = Stream(str_id, "rtsp://admin:sh123456@192.168.1.237:554/h264/ch1/main/av_stream")
        # src = Stream(str_id, "file:///media/lpr_test3.mp4")
        
        # uri = "rtsp://admin:sh123456@192.168.1.237:554/h264/ch1/main/av_stream"
        # src = Stream(id, uri)
        # print(uri)
        print(app.add_src(src))
        time.sleep(1)
        # app.play_src(str_id)

    # if id == 10:
    #     # print(app.del_src(str_id))
    #     # time.sleep(2)
    #     str_id = "task-{}".format(id)
    #     # src = Stream(str_id, "file:///media/test.ts")
    #     uri = "rtsp://admin:Hx123456@192.168.100.97:554/h264/ch1/main/av_stream"
    #     # src = Stream(str_id, "file:///media/LumphiniPark_HD_007.mp4")

    #     # uri = "rtsp://192.168.100.35:8554"
    #     src = Stream(str_id, uri)
    #     # print(uri)
    #     print(app.add_src(src))
    #     app.play_src(str_id)
    time.sleep(1)
    id = id + 1
   
    
    



# print(ret, msg)

# src = Stream(1, "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4")

# print(app.add_src(src))
# time.sleep(5)
# print("play over-------------------------")
# print(app.pause_src(src.id))
# time.sleep(10)
# print("pause over-------------------------")
# print(app.play_src(src.id))
# time.sleep(10)
# print("play over-------------------------")

# id = 1

# # src = Stream(1, "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4")

# # app.add_src(src)

# while True:
#     time.sleep(10)
#     src = Stream(id, "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4")
#     id = id + 1
#     app.add_src(src)
    
#     # app.add_src()
