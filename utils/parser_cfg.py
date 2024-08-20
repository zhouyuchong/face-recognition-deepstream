'''
Author: zhouyuchong
Date: 2024-08-19 14:54:53
Description: 
LastEditors: zhouyuchong
LastEditTime: 2024-08-20 10:03:25
'''
import os
import configparser
import toml
import numpy as np
def parse_args(cfg_path):
    cfg = toml.load(cfg_path)
    return cfg

def set_property(cfg, gst_element, name):
    properties = cfg[name]
    for key, value in properties.items():
        print(f"{name} set_property {key} {value} \n")
        gst_element.set_property(key, value)

def set_tracker_properties(tracker, path):
    config = configparser.ConfigParser()
    config.read(path)
    config.sections()

    for key in config['tracker']:
        if key == 'tracker-width' :
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height' :
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id' :
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file' :
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file' :
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)


def load_faces(path):
    loaded_faces = {}
    file_list = os.listdir(path)
    for file in file_list:
        if file.endswith('.npy'):
            face_feature = np.load(os.path.join(path, file)).reshape(-1, 1)
            name = file.split('.')[0]
            loaded_faces[name] = face_feature
    return loaded_faces