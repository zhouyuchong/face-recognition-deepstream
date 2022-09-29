# DSFace接口文档
## 1 - 服务启动
```
import init_path

from kbds.app import DSFace

app = DSFace("kafkaip:port:topic")
app.start()
```
## 2 - 接口说明
### 0. 
### 1. start()
+ 参数:
  + 无
+ 返回值：
  + True/False，状态信息(string)

    创建pipeline并进入NULL状态，等待输入
### 2. stop()
+ 参数：
  + 无
+ 返回值：
  + True/False，“success”(string)
  
    停止线程，设置pipeline进入NULL状态
### 3. add_src()
+ 参数:
  + 资源 [Stream](#3.1)
+ 返回值：
  + True/False， “状态信息”（String）
+ 异常：
  + exceed max source num：资源数超过/达到最大数量
  + source id exist：该资源名已经存在


### 4. get_src()
+ 参数：
  + 资源名称id
+ 返回值：
  + 资源相关信息

### 5. del_src()
+ 参数：
  + 资源名称id
+ 返回值：
  + 操作信息



## 3 - 对象说明
### 3.1 - Stream
+ 构造参数：
  + id, 字符串， 用户输入的资源名称
  + uri， 视频地址
+ 示例：
```
id = "hx-camera-01"
uri = "rtsp://admin:sh123456@192.168.1.237:554/h264/ch1/main/av_stream"
src = Stream(id, uri)
```


