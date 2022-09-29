import ctypes
import sys
sys.path.append('/opt/nvidia/deepstream/deepstream/lib')

def long_to_uint64(l):
    value = ctypes.c_uint64(l & 0xffffffffffffffff).value
    return value