# Kafka message format
## 1 - Face
### message m
+ m["object"]["id"]: deepstream id
+ m["object"]["face"]["cap"]: timestamp
+ m["object"]["face"]["gender"]: user src task id
+ m["object"]["face"]["glasses"]: face feature by arcface
+ m["object"]["face"]["facialhair"]: face image by retinaface
+ m["object"]["face"]["name"]: file name

## 3 - error bus
### topic：error
### message m
+ m["id"]: user source task id
+ m["message"]: error message details