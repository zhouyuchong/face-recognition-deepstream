#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, softwarenveglglessink
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import sys
sys.path.append('../')
import gi
import configparser
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
from gi.repository import GLib
from ctypes import *
import time
import sys
import math
import platform
from optparse import OptionParser
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.utils import long_to_uint64
from common.FPS import GETFPS

import pyds
import numpy as np
import cv2
import os
import os.path
from os import path
import json
import pickle 

import Gstpipeline.probes
import Gstpipeline.message

fps_streams={}

MAX_DISPLAY_LEN=64
MAX_TIME_STAMP_LEN = 32



MUXER_OUTPUT_WIDTH=1920
MUXER_OUTPUT_HEIGHT=1080
MUXER_BATCH_TIMEOUT_USEC=4000000
TILED_OUTPUT_WIDTH=1280
TILED_OUTPUT_HEIGHT=720
GST_CAPS_FEATURES_NVMM="memory:NVMM"
OSD_PROCESS_MODE= 0
OSD_DISPLAY_TEXT= 1
YOLO_PERSON_TRACK_ID = []
RFACE_TRACK_ID = []
AFACE_CAL_ID = []
CROPED_FACE = []


input_file = None
schema_type = 0
proto_lib = "../../../../lib/libnvds_kafka_proto.so"
conn_str = "localhost;9092;guest"
cfg_file = "config/cfg_kafka.txt"
topic = None
no_display = False

MSCONV_CONFIG_FILE = "config/dstest4_msgconv_config.txt"

def tiler_src_pad_buffer_probe(pad,info,u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        # frame_number=frame_meta.frame_num

        # Get frame rate through this probe
        fps_streams["stream{0}".format(frame_meta.pad_index)].get_fps()
        try:
            l_frame=l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def cb_newpad(decodebin, decoder_src_pad,data):
    print("In cb_newpad\n")
    caps=decoder_src_pad.get_current_caps()
    gststruct=caps.get_structure(0)
    gstname=gststruct.get_name()
    source_bin=data
    features=caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=",gstname)
    if(gstname.find("video")!=-1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        print("features=",features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad=source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")

def decodebin_child_added(child_proxy,Object,name,user_data):
    print("Decodebin child added:", name, "\n")
    if(name.find("decodebin") != -1):
        Object.connect("child-added",decodebin_child_added,user_data)

def create_source_bin(index,uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name="source-bin-%02d" %index
    print(bin_name)
    nbin=Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri",uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added",cb_newpad,nbin)
    uri_decode_bin.connect("child-added",decodebin_child_added,nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin,uri_decode_bin)
    bin_pad=nbin.add_pad(Gst.GhostPad.new_no_target("src",Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin

def main(args):
    global folder_name
    folder_name = "rface_out_crops"

    # Check input arguments
    if len(args) < 2:
        sys.stderr.write("usage: %s <uri1> [uri2] ... [uriN]\n" % args[0])
        sys.exit(1)

    for i in range(0,len(args)-1):
        fps_streams["stream{0}".format(i)]=GETFPS(i)
    number_sources=len(args)-1

    # registering callbacks
    pyds.register_user_copyfunc(Gstpipeline.message.meta_copy_func)
    pyds.register_user_releasefunc(Gstpipeline.message.meta_free_func)

    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    
    print("Creating streamux \n ")
    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)
    for i in range(number_sources):
        # os.mkdir(folder_name + "/stream_" + str(i))
        print("Creating source_bin ",i," \n ")
        uri_name=args[i+1]
        if uri_name.find("rtsp://") == 0 :
            is_live = True
        source_bin=create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)

        capname = "caps4src_%u"%i
        capname = Gst.ElementFactory.make("capsfilter", capname)
        capname.set_property('caps', Gst.Caps.from_string('video/x-raw(memory:NVMM'))
        pipeline.add(capname)

        videorate_name = "videorate_%u"%i
        print("Creating videorate ",i ," \n ")
        # Create videorate.
        videorate_name = Gst.ElementFactory.make("videorate", videorate_name)
        if not videorate_name:
            sys.stderr.write(" Unable to create videorate \n")
        videorate_name.set_property("max-rate", 40)
        videorate_name.set_property("drop-only", 1)
        pipeline.add(videorate_name)

        source_bin.link(capname)
        capname.link(videorate_name)
        # video_convertor_name.link(videorate_name)
        padname="sink_%u" %i
        sinkpad= streammux.get_request_pad(padname) 
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad=videorate_name.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)
        
    queue_streammux=Gst.ElementFactory.make("queue","queue_streammux")
    queue_pgie=Gst.ElementFactory.make("queue","queue_pgie")
    queue_tracker=Gst.ElementFactory.make("queue","queue_tracker")
    queue_tee_pgie_1=Gst.ElementFactory.make("queue","queue_tee_pgie_1")
    queue_tee_pgie_2=Gst.ElementFactory.make("queue","queue_tee_pgie_2")
    queue_tiler=Gst.ElementFactory.make("queue","queue_tiler")
    queue_vidconv=Gst.ElementFactory.make("queue","queue_vidconv")
    queue_osd=Gst.ElementFactory.make("queue","queue_osd")
    queue_sgie=Gst.ElementFactory.make("queue","queue_sgie")
    queue_tee_sgie_1=Gst.ElementFactory.make("queue","queue_tee_sgie_1")
    queue_tee_sgie_2=Gst.ElementFactory.make("queue","queue_tee_sgie_2")
    queue_msg_1=Gst.ElementFactory.make("queue","queue_msg1")
    queue_msg_2=Gst.ElementFactory.make("queue","queue_msg2")

    pipeline.add(queue_streammux)
    pipeline.add(queue_pgie)
    pipeline.add(queue_tracker)
    pipeline.add(queue_tee_pgie_1)
    pipeline.add(queue_tee_pgie_2)
    pipeline.add(queue_tiler)
    pipeline.add(queue_vidconv)
    pipeline.add(queue_osd)
    pipeline.add(queue_sgie)
    pipeline.add(queue_tee_sgie_2)
    pipeline.add(queue_tee_sgie_1)
    pipeline.add(queue_msg_1)
    pipeline.add(queue_msg_2)

    print("Creating Pgie \n ")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    print("Creating sgie \n ")
    sgie = Gst.ElementFactory.make("nvinfer", "Secondary-inference")
    if not sgie:
        sys.stderr.write(" Unable to create sgie \n")

    print("Creating tgie \n ")
    tgie = Gst.ElementFactory.make("nvinfer", "Third-inference")
    if not tgie:
        sys.stderr.write(" Unable to create tgie \n")

    print("Creating tracker \n ")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write(" Unable to create tracker \n")

    print("Creating pgie tee \n ")
    tee_pgie = Gst.ElementFactory.make("tee", "tee_pgie")
    if not tee_pgie:
        sys.stderr.write(" Unable to create tee_pgie \n")

    print("Creating sgie tee \n ")
    tee_sgie = Gst.ElementFactory.make("tee", "tee_sgie")
    if not tee_sgie:
        sys.stderr.write(" Unable to create tee_sgie \n")

    print("Creating sgie tee \n ")
    tee_msg = Gst.ElementFactory.make("tee", "tee_msg")
    if not tee_msg:
        sys.stderr.write(" Unable to create tee_sgie \n")

    print("Creating tiler \n ")
    tiler=Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")

    print("Creating tiler2 \n ")
    tiler2=Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler2")
    if not tiler2:
        sys.stderr.write(" Unable to create tiler2 \n")

    print("Creating nvvidconv \n ")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")

    print("Creating nvvidconv2 \n ")

    nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "convertor2")
    if not nvvidconv2:
        sys.stderr.write(" Unable to create nvvidconv2 \n")

    # Add nvvidconv1 and filter1 to convert the frames to RGBA
    # which is easier to work with in Python.
    print("Creating nvvidconv1 \n ")
    nvvidconv_rface = Gst.ElementFactory.make("nvvideoconvert", "retinaface-convertor")
    if not nvvidconv_rface:
        sys.stderr.write(" Unable to create nvvidconv1 \n ")

    print("Creating nvvidconv2 \n ")
    nvvidconv_rface2 = Gst.ElementFactory.make("nvvideoconvert", "retinaface-convertor2")
    if not nvvidconv_rface2:
        sys.stderr.write(" Unable to create nvvidconv2 \n ")
    
    print("Creating filter1 \n ")
    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter1 = Gst.ElementFactory.make("capsfilter", "filter1")
    if not filter1:
        sys.stderr.write(" Unable to get the caps filter1 \n")
    filter1.set_property("caps", caps1)

    print("Creating nvosd \n ")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")
    nvosd.set_property('process-mode',OSD_PROCESS_MODE)
    nvosd.set_property('display-text',OSD_DISPLAY_TEXT)
    
    print("Creating nvosd2 \n ")
    nvosd2 = Gst.ElementFactory.make("nvdsosd", "onscreendisplay2")
    if not nvosd2:
        sys.stderr.write(" Unable to create nvosd2 \n")
    nvosd2.set_property('process-mode',OSD_PROCESS_MODE)
    nvosd2.set_property('display-text',OSD_DISPLAY_TEXT)

    msgconv = Gst.ElementFactory.make("nvmsgconv", "nvmsg-converter")
    if not msgconv:
        sys.stderr.write(" Unable to create msgconv \n")

    msgbroker = Gst.ElementFactory.make("nvmsgbroker", "nvmsg-broker")
    if not msgbroker:
        sys.stderr.write(" Unable to create msgbroker \n")


    if(is_aarch64()):
        print("Creating transform \n ")
        transform=Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
        if not transform:
            sys.stderr.write(" Unable to create transform \n")

    print("Creating EGLSink \n")
    # sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    sink = Gst.ElementFactory.make("fakesink", "nvvideo-renderer")
    if not sink:
        sys.stderr.write(" Unable to create egl sink \n")

    print("Creating fakeSink \n")
    sink2 = Gst.ElementFactory.make("fakesink", "fakesink")
    if not sink2:
        sys.stderr.write(" Unable to create egl sink2 \n")

    print("Creating fakeSink \n")
    sink3 = Gst.ElementFactory.make("fakesink", "fakesink2")
    if not sink3:
        sys.stderr.write(" Unable to create egl sink3 \n")
            

    if is_live:
        print("Atleast one of the sources is live")
        streammux.set_property('live-source', 1)

    
    mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
    
    streammux.set_property("nvbuf-memory-type", mem_type)
    streammux.set_property('sync-inputs', 1)
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', number_sources)
    # 40000 performance better, don't know why
    streammux.set_property('batched-push-timeout', 4000)

    nvvidconv.set_property("nvbuf-memory-type", mem_type)
    nvvidconv2.set_property("nvbuf-memory-type", mem_type)

    pgie.set_property('config-file-path', "config/config_yolov5.txt")
    pgie_batch_size=pgie.get_property("batch-size")
    if(pgie_batch_size != number_sources):
        print("WARNING: Overriding infer-config batch-size",pgie_batch_size," with number of sources ", number_sources," \n")
        pgie.set_property("batch-size",number_sources)
    
    sgie.set_property('config-file-path', "config/config_retinaface.txt")
    tgie.set_property('config-file-path', "config/config_arcface.txt")

    msgconv.set_property('config', MSCONV_CONFIG_FILE)
    msgconv.set_property('payload-type', schema_type)
    msgbroker.set_property('proto-lib', proto_lib)
    msgbroker.set_property('conn-str', conn_str)
    if cfg_file is not None:
        msgbroker.set_property('config', cfg_file)
    if topic is not None:
        msgbroker.set_property('topic', topic)
    msgbroker.set_property('sync', False)

    tiler_rows=int(math.sqrt(number_sources))
    tiler_columns=int(math.ceil((1.0*number_sources)/tiler_rows))
    tiler.set_property("rows",tiler_rows)
    tiler.set_property("columns",tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)

    tiler2.set_property("nvbuf-memory-type", mem_type)
    nvvidconv_rface.set_property("nvbuf-memory-type", mem_type)

    tiler2.set_property("rows",tiler_rows)
    tiler2.set_property("columns",tiler_columns)
    tiler2.set_property("width", TILED_OUTPUT_WIDTH)
    tiler2.set_property("height", TILED_OUTPUT_HEIGHT)
    sink.set_property("qos",0)

    #Set properties of tracker
    config = configparser.ConfigParser()
    config.read('config/config_tracker.txt')
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
        if key == 'enable-batch-process' :
            tracker_enable_batch_process = config.getint('tracker', key)
            tracker.set_property('enable_batch_process', tracker_enable_batch_process)
        if key == 'enable-past-frame' :
            tracker_enable_past_frame = config.getint('tracker', key)
            tracker.set_property('enable_past_frame', tracker_enable_past_frame)

    print("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    pipeline.add(sgie)
    pipeline.add(tgie)
    pipeline.add(tracker)
    pipeline.add(tee_pgie)
    pipeline.add(tee_sgie)
    pipeline.add(tee_msg)
    pipeline.add(tiler)
    pipeline.add(tiler2)
    pipeline.add(nvvidconv)
    pipeline.add(nvvidconv2)
    pipeline.add(filter1)
    pipeline.add(nvvidconv_rface)
    pipeline.add(nvvidconv_rface2)
    pipeline.add(nvosd)
    pipeline.add(nvosd2)
    if is_aarch64():
        pipeline.add(transform)
    pipeline.add(sink)
    pipeline.add(sink2)
    pipeline.add(sink3)
    pipeline.add(msgconv)
    pipeline.add(msgbroker)

    print("Linking elements in the Pipeline \n")
    streammux.link(queue_streammux)
    queue_streammux.link(pgie)
    pgie.link(queue_pgie)
    queue_pgie.link(tracker)
    tracker.link(queue_tracker)
    queue_tracker.link(tee_pgie)
    tee_pgie.link(queue_tee_pgie_1)
    tee_pgie.link(queue_tee_pgie_2)

    queue_tee_pgie_1.link(tiler)
    tiler.link(queue_tiler)
    queue_tiler.link(nvvidconv)
    nvvidconv.link(queue_vidconv)
    queue_vidconv.link(nvosd)
    if is_aarch64():
        nvosd.link(queue_osd)
        queue_osd.link(transform)
        transform.link(sink)
    else:
        nvosd.link(queue_osd)
        queue_osd.link(sink)   

    queue_tee_pgie_2.link(sgie)
    sgie.link(queue_sgie)
    queue_sgie.link(tee_sgie)
    tee_sgie.link(queue_tee_sgie_1)
    queue_tee_sgie_1.link(tgie)
    tgie.link(nvvidconv2)
    nvvidconv2.link(sink2)
    
    tee_sgie.link(queue_tee_sgie_2)
    queue_tee_sgie_2.link(nvvidconv_rface)
    nvvidconv_rface.link(filter1)
    filter1.link(tiler2)
    # tiler2.link(sink3)
    tiler2.link(tee_msg)

    queue_msg_1.link(msgconv)
    msgconv.link(msgbroker)

    queue_msg_2.link(sink3)
    sink_pad = queue_msg_1.get_static_pad("sink")
    tee_msg_pad = tee_msg.get_request_pad('src_%u')
    tee_render_pad = tee_msg.get_request_pad("src_%u")
    if not tee_msg_pad or not tee_render_pad:
        sys.stderr.write("Unable to get request pads\n")
    tee_msg_pad.link(sink_pad)
    sink_pad = queue_msg_2.get_static_pad("sink")
    tee_render_pad.link(sink_pad)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)

    tiler_src_pad=pgie.get_static_pad("src")
    if not tiler_src_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        tiler_src_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_src_pad_buffer_probe, 0)
    

    # videorate_src_pad = streammux.get_static_pad("sink")
    # streammux_sink_pad.add_probe(Gst.PadProbeType.BUFFER, streammux_sink_pad_buffer_probe, 0)
    
    sgie_src_pad=queue_sgie.get_static_pad("sink")
    sgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, Gstpipeline.probes.sgie_src_pad_buffer_probe, 0)

    tgie_src_pad = nvvidconv2.get_static_pad("sink")
    tgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, Gstpipeline.probes.tgie_src_pad_buffer_probe, 0)

    picture_probe = tiler2.get_static_pad("sink")
    picture_probe.add_probe(Gst.PadProbeType.BUFFER, Gstpipeline.probes.pic_crop_probe, 0)
    # pipeline graph
    # graph_pipeline(pipeline)
    # List the sources
    print("Now playing...")
    for i, source in enumerate(args):
        if (i != 0):
            print(i, ": ", source)

    print("Starting pipeline \n")
    # start play back and listed to events		
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    print("Exiting app\n")
    pyds.unset_callback_funcs()

    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main(sys.argv))



