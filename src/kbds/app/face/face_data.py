import os
import threading
import time
from collections import deque

import numpy as np
import cv2
from requests import patch
import kbds.util.constant as constant
import kbds.util.picConv as convertor


MAX_FACE_IN_POOL = 40
TPOOL = set()

class FaceFeature():
    def __init__(self):
        self.link = ""
        self.state = constant.FaceState.Init
        self.bbox = None

    def set_frame_num(self, frame_num):
        self.frame_num = frame_num

    def set_source_id(self, source_id):
        self.source_id = source_id

    def set_timestamp(self, ts):
        self.ts = ts

    def set_bbox(self, bbox):
        self.bbox = bbox

    def set_state(self, state):
        self.state =  state

    def set_bg_image(self, array):
        self.bg_image = array

    def set_face_feature(self, array):
        self.face_feature = array

    def set_image_name(self, name):
        self.image_name = name

    def set_face_image_link(self, link):
        self.face_link = link

    def set_ff_link(self, link):
        self.ff_link = link

#== get functions=============================================================================

    def get_state(self):
        return self.state

    def get_bg_image(self):
        return self.bg_image

    def get_source_id(self):
        return self.source_id

    def get_frame_num(self):
        return self.frame_num

    def get_bbox(self):
        return self.bbox

    def get_ts(self):
        return self.ts

    def get_face_feature(self):
        return self.face_feature

    def get_image_name(self):
        return self.image_name

    def get_face_image_link(self):
        return self.face_link

    def get_ff_link(self):
        return self.ff_link

class FacePool():
    _instance = None
    lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance:
            return cls._instance
        else:
            with cls.lock:
                cls._instance = super().__new__(cls)
                return cls._instance

    def __init__(self):
        self.uninfered_face_num = 0
        self.pool = dict()        

    def add(self, id, face):
        self.pool[id] = face
        return True, "success"

    def id_exist(self, id):
        if id in self.pool:
            return True
        return False

    def counter(self, op):
        if op == constant.CounterOp.UP:
            self.uninfered_face_num = self.uninfered_face_num + 1
        if op == constant.CounterOp.DOWN:
            self.uninfered_face_num = self.uninfered_face_num - 1
            

    def check_full(self):
        if self.uninfered_face_num >= MAX_FACE_IN_POOL:
            return True
        else:
            return False
            
    def get_face_by_id(self, id):
        return self.pool[id]

    def pop_face_by_id(self, id):
        # print("pop id ", id)
        return self.pool.pop(id)

    def get_ids_in_pool(self):
        ids = self.pool.copy().keys()
        return ids

    def check_msg_status(self, id):
        if self.pool[id].get_state() == constant.FaceState.State3:
            self.pool[id].set_state(constant.FaceState.State4)
            return self.pool[id].get_face_image_link(), self.pool[id].get_ff_link(), self.pool[id].get_image_name(), self.pool[id].get_ts(), self.pool[id].get_source_id()

    def check_and_save(self):
        tmp_pool = self.pool.copy()
        for id in tmp_pool:
            if tmp_pool[id].get_state() == constant.FaceState.State2:
                ret2 = self.save_face_to_local(id=id, face=tmp_pool[id])
                ret3 = self.save_face_feature_to_local(id=id, face=tmp_pool[id])
                if ret2 and ret3:
                    self.pool[id].set_state(constant.FaceState.State3)

    def save_face_to_local(self, id, face):
        img_path = "images/origin/face-{}.png".format(id)
        if os.path.exists("images/origin/face-{}.png".format(id)):
            name = "face-{}.png".format(id)
            face.set_image_name(name)
            print("face image save to :", img_path)
            face.set_face_image_link(img_path)          
            return True
        else:
            return False

    def save_face_feature_to_local(self, id, face):   
        ff = face.get_face_feature()
        ff_path = "images/face_feature/face-{0}-{1}.npy".format(id, face.get_ts())
        path = 'images/face_feature'
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(ff_path, ff)
        print("face feature save to :", ff_path)
        face.set_ff_link(ff_path)
        
        return True

 
class TerminatePool():
    def add(self, id):
        TPOOL.add(id)
    
    def id_exist(self, id):
        if id in TPOOL:
            return True
        return False


class Facethread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.daemon = True
        self._lock = threading.Lock()
        self._timeout = 30
        self.num_in_pool = 0
        self.face_pool = FacePool()
        self.create_save_dir()
        
    def create_save_dir(self):
        path = "images/origin"
        if not os.path.exists(path):
            os.makedirs(path)
        path = "images/aligned"
        if not os.path.exists(path):
            os.makedirs(path)
          

    def run(self):
        while True:
            time.sleep(1)
            self.face_pool.check_and_save()
