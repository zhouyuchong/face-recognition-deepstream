import init_path

from kbds.app import DSFace
from kbds import Stream
import traceback
import time

app = DSFace("localhost;9092;deepstream")
print("--------- start app")
print(app.start())
time.sleep(2)

print("-----------------------------------------------------------------------------")
id = 0
while True:
    if id == 0:
        str_id = "task-{}".format(id)
        src = Stream(str_id, "file:///media/test.ts")
        print(app.add_src(src))
        time.sleep(1)
        app.play_src(str_id)

    if id == 5:
        str_id = "task-{}".format(id)
        src = Stream(str_id, "file:///media/test.ts")
        print(app.add_src(src))
        app.play_src(str_id)

    if id == 10:
        print("get status of task-5: ", app.get_src(id="task-5"))

        print("get status of task-1111: ", app.get_src(id="task-1111"))

    if id == 15:
        print("======== delete src test =======")
        app.del_src(id="task-0")

    if id == 20:
        print("======== delete src test =======")
        app.del_src(id="task-5")
        
    if id == 25:
        str_id = "task-{}".format(5)
        src = Stream(str_id, "file:///media/test.ts")
        print(app.add_src(src))
        app.play_src(str_id)

    if id == 30:
        print("======== delete src test =======")
        app.del_src(id="task-5")
    time.sleep(1)
    id = id + 1
   

