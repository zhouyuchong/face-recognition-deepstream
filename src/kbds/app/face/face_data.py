from multiprocessing.connection import wait
import os
import threading
import time
from collections import deque

import numpy as np
import cv2
from requests import patch
import kbds.util.constant as constant
import kbds.util.picConv as convertor
from kbds.util.fdfs_util.fdfs_util import FastDfsUtil


MAX_FACE_IN_POOL = 40
FACE_POOL = {}
TPOOL = set()

class FaceFeature():
    def __init__(self):
        self.link = ""
        self.state = constant.FaceState.Init

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
    def __init__(self):
        self.uninfer_face = 0

    def add(self, id, face):
        FACE_POOL[id] = face
        return True, "success"

    def id_exist(self, id):
        if id in FACE_POOL:
            return True
        return False

    def counter(self, op):
        if op == constant.CounterOp.UP:
            self.uninfer_face = self.uninfer_face + 1
        if op == constant.CounterOp.DOWN:
            self.uninfer_face = self.uninfer_face - 1
            

    def check_full(self):
        if self.uninfer_face >= MAX_FACE_IN_POOL:
            return True
        else:
            return False
            
    def get_face_by_id(self, id):
        return FACE_POOL[id]

    def pop_face_by_id(self, id):
        # print("pop id ", id)
        return FACE_POOL.pop(id)

    def get_pool(self):
        return FACE_POOL
 
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
        self.create_save_dir()
        

    def create_save_dir(self):
        path = "images/origin"
        if not os.path.exists(path):
            os.makedirs(path)
        path = "images/aligned"
        if not os.path.exists(path):
            os.makedirs(path)

    def check_new(self):
        if not self._lock.acquire(timeout=self._timeout):
            print("Fail to acquire lock, maybe busy")
            return False

        for i in FACE_POOL.copy():
            if i in FACE_POOL.copy():
                if FACE_POOL.copy()[i].get_state() == constant.FaceState.State2:
                    # print(FACE_POOL)
                    # ret1 = self.save_bg_to_local(id=i, face=FACE_POOL[i])
                    ret2 = self.save_face_to_local(id=i, face=FACE_POOL[i])
                    ret3 = self.save_face_feature_to_local(id=i, face=FACE_POOL[i])
                    
                    FACE_POOL[i].set_state(constant.FaceState.State3)
 
        self._lock.release()
          
    #== save functions=============================================================================

    def save_bg_to_local(self, id, face):
        # print("save background {} image to local.".format(id))
        # temp = FastDfsUtil()
        bg = np.array(face.get_bg_image(), copy=True, order='C')
        bg = cv2.cvtColor(bg, cv2.COLOR_RGBA2BGRA)
        path = 'images/background'
        img_path = "images/background/backgrond-of-face-{0}.png".format(id, face.get_ts())
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(img_path, bg)
        # temp.upload_by_filename(img_path)
        return True

    def save_face_to_local(self, id, face):
        # print("save face {} to local.".format(id))
        # path = 'save/face'
        # if not os.path.exists(path):
        #     os.makedirs(path)
        img_path = "images/origin/face-{}.png".format(id)
        os.path.exists("images/origin/face-{}.png".format(id))
        # frame_copy = face.get_bbox()
        # img_path = "save/face/{0}-{1}-{2}.jpg".format(id, face.get_source_id(), face.get_ts())
        # cv2.imwrite(img_path, frame_copy)
        name = "face-{}.png".format(id)
        face.set_image_name(name)
        # face.set_face_image_link(img_path)
        save_fdfs = FastDfsUtil()      
        ret = save_fdfs.upload_by_filename(img_path)
        save_p = ret["Remote file_id"].decode('utf-8')
        # save_p = "test"
        print(save_p)
        face.set_face_image_link(save_p)          
        return True

    def save_face_feature_to_local(self, id, face):   
        # print("save feature {} to local.".format(id))
        ff = face.get_face_feature()
        ff_path = "images/face_feature/face-{0}-{1}.npy".format(id, face.get_ts())
        # face.set_ff_link(ff_path)
        path = 'images/face_feature'
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(ff_path, ff)

        save_fdfs = FastDfsUtil()      
        ret = save_fdfs.upload_by_filename(ff_path)
        save_p = ret["Remote file_id"].decode('utf-8')
        # save_p = "test"
        face.set_ff_link(save_p)
        
        return True
    
    def run(self):
        while True:
            time.sleep(1)
            self.check_new()
