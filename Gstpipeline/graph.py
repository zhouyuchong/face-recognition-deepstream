import sys
sys.path.append('../')
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
from gi.repository import GLib
import os

default_png_path = "/tmp/pipeline.png"
default_dot_path = "/tmp/pipeline.dot"

def graph_pipeline(pipeline, pngpath=default_png_path, dotpath=default_dot_path):
    Gst.debug_bin_to_dot_file(pipeline, Gst.DebugGraphDetails.ALL, "pipeline")
    try:
        os.system("dot -Tpng -o {} {}".format(pngpath, dotpath))
    except Exception:
        print("error")