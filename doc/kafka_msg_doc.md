# Kafka 消息格式说明
## 1 - Face
### 主体消息m
+ m["object"]["id"]: 算法跟踪的id
+ m["object"]["face"]["cap"]: 时间戳
+ m["object"]["face"]["gender"]: 视频源的id(输入时的视频源名称)
+ m["object"]["face"]["glasses"]: 脸部特征文件(.npy文件)
+ m["object"]["face"]["facialhair"]: fastfds人脸截图链接
+ m["object"]["face"]["name"]: 人脸截图图片文件名名称(*.jpg)

## 2 - LPR
### 主体消息m
+ m["object"]["id"]: 算法跟踪的id
+ m["object"]["face"]["cap"]: 时间戳
+ m["object"]["face"]["gender"]: 视频源的id(输入时的视频源名称)
+ m["object"]["face"]["glasses"]: 脸部特征文件(.npy文件)
+ m["object"]["face"]["facialhair"]: fastfds人脸截图链接
+ m["object"]["face"]["name"]: 人脸截图图片文件名名称(*.jpg)

## 3 - 异常
### 话题：error
### 主体消息m
+ m["id"]: 视频源id
+ m["message"]: 错误信息