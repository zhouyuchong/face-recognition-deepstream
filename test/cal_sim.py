from importlib.resources import path
import numpy as np
import os

def takeFirst(elem):
    return elem[0]

check_face = np.load('./images/face_feature/face-0-1664442738.836995.npy')

check_face = check_face.reshape(1, 512)
face = []
dirpath = './images/face_feature/'
face = os.listdir(dirpath)
face_list = []
for f in face:
    tface = np.load(dirpath + f)
    re = check_face.dot(tface)
    face_list.append([re[0][0], f])
    
face_list.sort(key=takeFirst, reverse=True)
for j in range(len(face_list)):
    print(face_list[j])   
