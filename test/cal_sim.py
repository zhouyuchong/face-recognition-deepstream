from importlib.resources import path
import numpy as np
import os

check_face = np.load('./save/face_feature/0-0-2022092709:32:59.npy')
# check_face = np.load('./alignmentface-of-face-0-frame-0_trt.npy')

check_face = check_face.reshape(1, 512)
print(check_face)

face = np.load('./save/face_feature/cpu.npy')
re = check_face.dot(face)
print(re)

# # print(check_face.shape)
# face = []
# dirpath = './save/face_feature/'
# face = os.listdir(dirpath)
# # print(face)
# for f in face:
#     tface = np.load(dirpath + f)
#     # print(tface.shape)
#     # re = check_face.dot(tface)
#     # print(re, f)
#     print(tface)
# face = np.load(npypath)
