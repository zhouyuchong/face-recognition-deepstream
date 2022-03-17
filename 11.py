import json

a = 1
print(a, type(a))
str = json.dumps([2,3,4])
a = str
print(a, type(a))