import kbds

import sys
import os
import configparser
import gi

from kbds.core import pipeline
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import time
import pyds

from kbds import sysp
from kbds import DSPipeline
import kbds.core.srcm
from kbds.util.FPS import PERF_DATA
from kbds.util.format import long_to_uint64
from kbds.util.picConv import crop_object
from kbds.app.face.face_data import Facethread
import kbds.util.constant as constant
from .face_data import *
import ctypes
ctypes.cdll.LoadLibrary('/opt/models/retinaface/libplugin_rface.so')

APP_NAME = "face"
APP_CONFIG_FOLDER = os.path.join(sysp.cfg_path, APP_NAME)
APP_MODEL_FOLDER = os.path.join(sysp.model_path, APP_NAME)
APP_LIB_FOLDER = os.path.join(sysp.lib_path, APP_NAME)


GPU_ID = 0
MAX_NUM_SOURCES = 16
mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)

schema_type = 0
proto_lib = "/opt/nvidia/deepstream/deepstream-6.1/lib/libnvds_kafka_proto.so"

# conn_str = "localhost;19092;deepstream"

MUXER_OUTPUT_WIDTH=1920
MUXER_OUTPUT_HEIGHT=1080
TILED_OUTPUT_WIDTH=1280
TILED_OUTPUT_HEIGHT=720
MAX_TIME_STAMP_LEN = 32


PGIE_CONFIG_FILE = os.path.join(sysp.cfg_path, APP_NAME, "config_retinaface.txt")
SGIE_CONFIG_FILE = os.path.join(sysp.cfg_path, APP_NAME, "config_arcface.txt")
TRACKER_CONFIG_FILE = os.path.join(sysp.cfg_path, APP_NAME, "config_tracker.txt")
MSCONV_CONFIG_FILE = os.path.join(sysp.cfg_path, APP_NAME, "config_msgconv.txt")
MSGBROKER_CONFIG_FILE = os.path.join(sysp.cfg_path, APP_NAME, "config_kafka.txt")

perf_data = None
is_first = True

face_pool = FacePool()
tpool = TerminatePool()

class DSFace(DSPipeline):

    def __init__(self, conn_str) -> None:
        super().__init__(conn_str=conn_str)
        self.conn_str = conn_str
        t = Facethread()
        t.start()

    def build_pipeline_abs(self, pipeline, streammux, msgconv, msgbroker, analytics):
        """
        build pipeline abstract methodn_frame = crop_object(n_frame, obj_meta)
        :param pipeline: Gst.Pipeline, stream pipeline
        :param streammux: Gst.Element, nvstreammux
        :param msgconv: Gst.Element, nvmsgconv
        :param msgbroker: Gst.Element, nvmsgbroker
        :return: (bool, message), result & message
        """
        # FPS counter
        global perf_data
        perf_data = PERF_DATA(MAX_NUM_SOURCES)

        # pgie
        self.pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        if not self.pgie:
            return False, "unable to create pgie"

        # tracker
        self.tracker = Gst.ElementFactory.make("nvtracker", "tracker")
        if not self.tracker:
            return False, "unable to cretae tracker"

        # tiler
        self.tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
        if not self.tiler:
            return False, "unable to create tiler"

        # nvvideoconvert
        self.nvvideoconvert = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        if not self.nvvideoconvert:
            return False, "unable to create nvvideoconvert"     

        # filter
        caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
        self.filter = Gst.ElementFactory.make("capsfilter", "filter")
        if not self.filter:
            sys.stderr.write(" Unable to get the caps filter \n")
        self.filter.set_property("caps", caps)

        # nvosd
        self.nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        if not self.nvosd:
            return False, "unable to create nvdsosd"   

        # sgie
        self.sgie = Gst.ElementFactory.make("nvinfer", "secondary-nvinference-engine")
        if not self.sgie:
            return False, "uable to make sgie"

        self.tee = Gst.ElementFactory.make("tee", "nvsink-tee")
        if not self.tee:
            return False, " Unable to create tee"

        # self.sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        self.sink = Gst.ElementFactory.make("fakevideosink", "nvvideo-renderer")
        if not self.sink:
            return False, "unable to create egl sink"

        # queues to avoid getting stuck
        self.queue1=Gst.ElementFactory.make("queue","queue1")
        self.queue2=Gst.ElementFactory.make("queue","queue2")
        self.queue3=Gst.ElementFactory.make("queue","queue3")
        self.queue4=Gst.ElementFactory.make("queue","queue4")
        self.queue5=Gst.ElementFactory.make("queue","queue5")
        self.queue6=Gst.ElementFactory.make("queue","queue6")
        self.queue7=Gst.ElementFactory.make("queue","queue7")
        self.queue8=Gst.ElementFactory.make("queue","queue8")
        self.queue_msg=Gst.ElementFactory.make("queue","queue_msg")
        self.queue_sink=Gst.ElementFactory.make("queue","queue_sink")
        pipeline.add(self.queue1)
        pipeline.add(self.queue2)
        pipeline.add(self.queue3)
        pipeline.add(self.queue4)
        pipeline.add(self.queue5)
        pipeline.add(self.queue6)
        pipeline.add(self.queue7)
        pipeline.add(self.queue8)
        pipeline.add(self.queue_sink)
        pipeline.add(self.queue_msg)

        # config pipeline
        streammux.set_property("nvbuf-memory-type", mem_type)
        streammux.set_property('width', constant.MUXER_OUTPUT_WIDTH)
        streammux.set_property('height', constant.MUXER_OUTPUT_HEIGHT)

        self.pgie.set_property('config-file-path', PGIE_CONFIG_FILE)
        self.sgie.set_property('config-file-path', SGIE_CONFIG_FILE)

        config = configparser.ConfigParser()
        config.read(TRACKER_CONFIG_FILE)
        config.sections()

        for key in config['tracker']:
            if key == 'tracker-width' :
                tracker_width = config.getint('tracker', key)
                self.tracker.set_property('tracker-width', tracker_width)
            if key == 'tracker-height' :
                tracker_height = config.getint('tracker', key)
                self.tracker.set_property('tracker-height', tracker_height)
            if key == 'gpu-id' :
                tracker_gpu_id = config.getint('tracker', key)
                self.tracker.set_property('gpu_id', tracker_gpu_id)
            if key == 'll-lib-file' :
                tracker_ll_lib_file = config.get('tracker', key)
                self.tracker.set_property('ll-lib-file', tracker_ll_lib_file)
            if key == 'll-config-file' :
                tracker_ll_config_file = os.path.join(sysp.cfg_path, config.get('tracker', key))
                self.tracker.set_property('ll-config-file', tracker_ll_config_file)
            if key == 'enable-batch-process' :
                tracker_enable_batch_process = config.getint('tracker', key)
                self.tracker.set_property('enable_batch_process', tracker_enable_batch_process)        

        self.pgie.set_property("batch-size", MAX_NUM_SOURCES)
        self.pgie.set_property("gpu_id", GPU_ID)   

        self.tiler.set_property("width", TILED_OUTPUT_WIDTH)
        self.tiler.set_property("height", TILED_OUTPUT_HEIGHT) 
        self.tiler.set_property("gpu_id", GPU_ID)

        self.nvvideoconvert.set_property("gpu_id", GPU_ID)
        self.nvvideoconvert.set_property("nvbuf-memory-type", mem_type)
        self.nvosd.set_property("gpu_id", GPU_ID)  

        # add elements to pipeline
        pipeline.add(self.pgie)
        pipeline.add(self.sgie)
        pipeline.add(self.tracker)
        pipeline.add(self.tiler)
        pipeline.add(self.nvvideoconvert)
        pipeline.add(self.filter)
        pipeline.add(self.tee)
        pipeline.add(self.nvosd)
        pipeline.add(self.sink)  
        pipeline.add(msgconv)
        pipeline.add(msgbroker)   

        # linking elments in the pipeline
        streammux.link(self.queue1)
        self.queue1.link(self.pgie)
        self.pgie.link(self.queue2)
        self.queue2.link(self.tracker)
        self.tracker.link(self.queue3)
        self.queue3.link(self.nvvideoconvert)
        self.nvvideoconvert.link(self.queue4)
        self.queue4.link(self.filter)
        self.filter.link(self.queue5)
        self.queue5.link(self.sgie)
        self.sgie.link(self.queue6)
          
        self.queue6.link(self.tee)
        self.queue_msg.link(msgconv)
        msgconv.link(msgbroker)
        self.queue_sink.link(self.sink)

        sink_pad = self.queue_msg.get_static_pad("sink")
        tee_msg_pad = self.tee.get_request_pad('src_%u')
        tee_render_pad = self.tee.get_request_pad("src_%u")
        if not tee_msg_pad or not tee_render_pad:
            sys.stderr.write("Unable to get request pads\n")
        tee_msg_pad.link(sink_pad)
        sink_pad = self.queue_sink.get_static_pad("sink")
        tee_render_pad.link(sink_pad)

        self.sink.set_property("sync", 0)
        self.sink.set_property("qos",0)
        msgconv.set_property('config', MSCONV_CONFIG_FILE)
        msgconv.set_property('payload-type', schema_type)
        
        msgbroker.set_property('proto-lib', proto_lib)
        print(self.conn_str)
        msgbroker.set_property('conn-str', self.conn_str)
        if MSGBROKER_CONFIG_FILE is not None:
            msgbroker.set_property('config', MSGBROKER_CONFIG_FILE)
        
        msgbroker.set_property('sync', False)

        # add probes to pipeline
        self.pgie_src_pad=self.queue5.get_static_pad("sink")
        if not self.pgie_src_pad:
            sys.stderr.write(" Unable to get src pad \n")
        else:
            self.pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, DSFaceProbe.pgie_sink_pad_buffer_probe, 0)
            GLib.timeout_add(5000, perf_data.perf_print_callback)

        self.sgie_sink_pad = self.queue6.get_static_pad("sink")
        if not self.sgie_sink_pad:
            return False, "fail"
        else:
            self.sgie_sink_pad.add_probe(Gst.PadProbeType.BUFFER, DSFaceProbe.sgie_sink_pad_buffer_probe, 0)


        self.msg_sink_pad = self.queue6.get_static_pad("sink")
        if not self.msg_sink_pad:
            return False, "fail"
        else:
            self.msg_sink_pad.add_probe(Gst.PadProbeType.BUFFER, DSFaceProbe.msg_sink_pad_buffer_probe, self.srcm)

        return True, "success"

class DSFaceProbe(DSFace):
    
    def pgie_sink_pad_buffer_probe(pad,info,u_data):
        """
        Probe to extract facial info from metadata and add them to Face pool. 
        
        Should be after retinaface.
        """
        global face_pool, tpool
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
            
            l_obj=frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    # Casting l_obj.data to pyds.NvDsObjectMeta
                    #obj_meta=pyds.glist_get_nvds_object_meta(l_obj.data)
                    obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                
                
                drop_signal = True
                tmp_face_bbox = [obj_meta.rect_params.width, obj_meta.rect_params.height]

                # if this face already exist as well as there still space for arcface to infer
                # check its states and bbox info
                if face_pool.id_exist(obj_meta.object_id) and face_pool.check_full() is False:
                    tmp_face = face_pool.get_face_by_id(obj_meta.object_id)
                    # this face has already sent a message
                    if tmp_face.get_state() == constant.FaceState.State4:
                        origin_bbox = tmp_face.get_bbox()
                        if (tmp_face_bbox[0] - origin_bbox[0]) > 15 and (tmp_face_bbox[1] - origin_bbox[1]) > 15:
                            print("detect face-{} with higher resolution".format(obj_meta.object_id), origin_bbox, " -->", tmp_face_bbox)
                            tmp_face.set_bbox(tmp_face_bbox)
                            tmp_face.set_state(constant.FaceState.State1)
                            face_pool.counter(op='up')
                            drop_signal = False

                if face_pool.id_exist(obj_meta.object_id) == False \
                    and tpool.id_exist(obj_meta.object_id) == False \
                    and face_pool.check_full() == False:
                    face = FaceFeature()
                    ts = time.time()
                    face.set_frame_num(frame_num=frame_meta.frame_num)
                    face.set_source_id(source_id=frame_meta.source_id)
                    face.set_timestamp(ts=ts)
                    # n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                    # face.set_bg_image(array=n_frame)
                    
                    # n_frame = crop_object(n_frame, [obj_meta.rect_params.top, obj_meta.rect_params.left, \
                        # obj_meta.rect_params.width, obj_meta.rect_params.height])
                    # frame_copy = np.array(n_frame, copy=True, order='C')
                    # convert the array into cv2 default color format
                    # frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGRA)
                        
                    face.set_bbox(tmp_face_bbox)
                    face.set_state(state=constant.FaceState.State1)
                    face_pool.add(id=obj_meta.object_id, face=face)
                    print("detect face-{} with resolution {}x{}".format(obj_meta.object_id, int(tmp_face_bbox[0]), int(tmp_face_bbox[1])))
                    face_pool.counter(op='up')
                    drop_signal = False

          
                try: 
                    l_obj=l_obj.next
                    if drop_signal is True:
                        pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)
                    
                except StopIteration:
                    break
            stream_index = "stream{0}".format(frame_meta.pad_index)
            global perf_data
            perf_data.update_fps(stream_index)
            # Get frame rate through this probe
            try:
                l_frame=l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def sgie_sink_pad_buffer_probe(pad,info,u_data):
        """
        Probe to extract facial feature from user-meta data. 
        
        Should be after arcface.
        """
        global face_pool
        frame_number=0
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
                # The casting is done by pyds.glist_get_nvds_frame_meta()
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone.
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break
            
            l_obj=frame_meta.obj_meta_list

            while l_obj is not None:
                try:
                    # Casting l_obj.data to pyds.NvDsObjectMeta
                    obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
                    
                except StopIteration:
                    break

                if face_pool.id_exist(obj_meta.object_id):
                    temp_face = face_pool.get_face_by_id(obj_meta.object_id)
                    if temp_face.get_state() == constant.FaceState.State1:
                        l_user_meta = obj_meta.obj_user_meta_list
                        while l_user_meta:
                            try:
                                user_meta = pyds.NvDsUserMeta.cast(l_user_meta.data) 
                            except StopIteration:
                                break
                            if user_meta and user_meta.base_meta.meta_type==pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META: 
                                try:
                                    tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                                except StopIteration:
                                    break
                        
                                layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                                output = []
                                for i in range(512):
                                    output.append(pyds.get_detections(layer.buffer, i))
                                res = np.reshape(output,(512,-1))
                                norm=np.linalg.norm(res)                    
                                normal_array = res / norm
                                temp_face.set_face_feature(normal_array)
                                temp_face.set_state(constant.FaceState.State2)
                                # print("get face {} feature".format(obj_meta.object_id))
                                try:
                                    l_user_meta = l_user_meta.next
                                except StopIteration:
                                    break
                        face_pool.counter(op='down')
                try: 
                    l_obj=l_obj.next

                except StopIteration:
                    break  
            try:
                l_frame=l_frame.next
            except StopIteration:
                break
                
        return Gst.PadProbeReturn.OK	

    def msg_sink_pad_buffer_probe(pad,info,srcm):
        """
        probe to add info into msg-meta data.
        """
        global face_pool
        global tpool
        

        frame_number=0
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
                # The casting is done by pyds.glist_get_nvds_frame_meta()
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone.
                #frame_meta = pyds.glist_get_nvds_frame_meta(l_frame.data)
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break
            
            frame_number=frame_meta.frame_num
            
            ids = face_pool.get_ids_in_pool()
            for id in ids:
                tmp_face = face_pool.get_face_by_id(id=id)
                if tmp_face.get_state() == constant.FaceState.State3:
                    image_link, ff_link, name, ts, sid = face_pool.check_msg_status(id=id)
                    uid = srcm.get_id_by_idx(sid)
                    face_pool.pop_face_by_id(id)
                    tpool.add(id)
                    msg_meta = pyds.alloc_nvds_event_msg_meta()
                    msg_meta.bbox.top = 0
                    msg_meta.bbox.left = 0
                    msg_meta.bbox.width = 0
                    msg_meta.bbox.height = 0
                    msg_meta.frameId = frame_number
                    msg_meta.trackingId = long_to_uint64(id)
                    msg_meta.confidence = 0
                    msg_meta = DSFaceMessage.generate_event_msg_meta(msg_meta, image_link, ff_link, uid, name, ts)
                    user_event_meta = pyds.nvds_acquire_user_meta_from_pool(batch_meta)
                
                    if user_event_meta:
                        user_event_meta.user_meta_data = msg_meta
                        user_event_meta.base_meta.meta_type = pyds.NvDsMetaType.NVDS_EVENT_MSG_META
                        # Setting callbacks in the event msg meta. The bindings
                        # layer will wrap these callables in C functions.
                        # Currently only one set of callbacks is supported.
                        pyds.user_copyfunc(user_event_meta, DSFaceMessage.meta_copy_func)
                        pyds.user_releasefunc(user_event_meta, DSFaceMessage.meta_free_func)
                        pyds.nvds_add_user_meta_to_frame(frame_meta,
                                                        user_event_meta)
                    else:
                        print("Error in attaching event meta to buffer\n")

      
            try:
                l_frame=l_frame.next
            except StopIteration:
                break
                
        return Gst.PadProbeReturn.OK	

    def sgie_sink_pad_buffer_probe_2(pad,info,u_data):
        """
        Probe to extract facial feature from user-meta data. 
        
        Should be after arcface.
        """
        global face_pool
        frame_number=0
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
                # The casting is done by pyds.glist_get_nvds_frame_meta()
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone.
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break
            
          
            try:
                l_frame=l_frame.next
            except StopIteration:
                break
                
        return Gst.PadProbeReturn.OK	


class DSFaceMessage(object):
        
    # Callback function for deep-copying an NvDsEventMsgMeta struct
    def meta_copy_func(data, user_data):
        # Cast data to pyds.NvDsUserMeta
        user_meta = pyds.NvDsUserMeta.cast(data)
        src_meta_data = user_meta.user_meta_data
        # Cast src_meta_data to pyds.NvDsEventMsgMeta
        srcmeta = pyds.NvDsEventMsgMeta.cast(src_meta_data)
        # Duplicate the memory contents of srcmeta to dstmeta
        # First use pyds.get_ptr() to get the C address of srcmeta, then
        # use pyds.memdup() to allocate dstmeta and copy srcmeta into it.
        # pyds.memdup returns C address of the allocated duplicate.
        dstmeta_ptr = pyds.memdup(pyds.get_ptr(srcmeta),
                                sys.getsizeof(pyds.NvDsEventMsgMeta))
        # Cast the duplicated memory to pyds.NvDsEventMsgMeta
        dstmeta = pyds.NvDsEventMsgMeta.cast(dstmeta_ptr)

        # Duplicate contents of ts field. Note that reading srcmeat.ts
        # returns its C address. This allows to memory operations to be
        # performed on it.
        dstmeta.ts = pyds.memdup(srcmeta.ts, MAX_TIME_STAMP_LEN + 1)

        # Copy the sensorStr. This field is a string property. The getter (read)
        # returns its C address. The setter (write) takes string as input,
        # allocates a string buffer and copies the input string into it.
        # pyds.get_string() takes C address of a string and returns the reference
        # to a string object and the assignment inside the binder copies content.
        dstmeta.sensorStr = pyds.get_string(srcmeta.sensorStr)

        if srcmeta.objSignature.size > 0:
            dstmeta.objSignature.signature = pyds.memdup(
                srcmeta.objSignature.signature, srcmeta.objSignature.size)
            dstmeta.objSignature.size = srcmeta.objSignature.size

        if srcmeta.extMsgSize > 0:
            if srcmeta.objType == pyds.NvDsObjectType.NVDS_OBJECT_TYPE_FACE:
                srcobj = pyds.NvDsFaceObject.cast(srcmeta.extMsg)
                obj = pyds.alloc_nvds_face_object()
                # obj.faceimage = pyds.get_string(srcobj.faceimage)
                # obj.feature = pyds.get_string(srcobj.feature)
                obj.age = srcobj.age
                obj.gender = pyds.get_string(srcobj.gender)
                obj.cap = pyds.get_string(srcobj.cap)
                obj.hair = pyds.get_string(srcobj.hair)
                obj.glasses = pyds.get_string(srcobj.glasses)
                obj.facialhair = pyds.get_string(srcobj.facialhair)
                obj.name = pyds.get_string(srcobj.name)
                obj.eyecolor = pyds.get_string(srcobj.eyecolor)
                dstmeta.extMsg = obj
                dstmeta.extMsgSize = sys.getsizeof(pyds.NvDsFaceObject)

        return dstmeta

    # Callback function for freeing an NvDsEventMsgMeta instance
    def meta_free_func(data, user_data):
        user_meta = pyds.NvDsUserMeta.cast(data)
        srcmeta = pyds.NvDsEventMsgMeta.cast(user_meta.user_meta_data)

        # pyds.free_buffer takes C address of a buffer and frees the memory
        # It's a NOP if the address is NULL
        pyds.free_buffer(srcmeta.ts)
        pyds.free_buffer(srcmeta.sensorStr)

        if srcmeta.objSignature.size > 0:
            pyds.free_buffer(srcmeta.objSignature.signature)
            srcmeta.objSignature.size = 0

        if srcmeta.extMsgSize > 0:
            if srcmeta.objType == pyds.NvDsObjectType.NVDS_OBJECT_TYPE_FACE:
                obj = pyds.NvDsFaceObject.cast(srcmeta.extMsg)
                # pyds.free_buffer(obj.faceimage)
                # pyds.free_buffer(obj.feature)
                pyds.free_buffer(obj.gender)
                pyds.free_buffer(obj.hair)
                pyds.free_buffer(obj.cap)
                pyds.free_buffer(obj.glasses)
                pyds.free_buffer(obj.facialhair)
                pyds.free_buffer(obj.name)
                pyds.free_buffer(obj.eyecolor)

            pyds.free_gbuffer(srcmeta.extMsg)
            srcmeta.extMsgSize = 0

    def generate_event_msg_meta(data, image_link, ff_link, id, name, ts):
        meta = pyds.NvDsEventMsgMeta.cast(data)
        meta.sensorId = 0
        meta.placeId = 0
        meta.moduleId = 0
        meta.sensorStr = "sensor-0"
        meta.ts = pyds.alloc_buffer(MAX_TIME_STAMP_LEN + 1)
        pyds.generate_ts_rfc3339(meta.ts, MAX_TIME_STAMP_LEN)

        # This demonstrates how to attach custom objects.
        # Any custom object as per requirement can be generated and attached
        # like NvDsVehicleObject / NvDsPersonObject. Then that object should
        # be handled in payload generator library (nvmsgconv.cpp) accordingly.
        
        meta.type = pyds.NvDsEventType.NVDS_EVENT_ENTRY
        meta.objType = pyds.NvDsObjectType.NVDS_OBJECT_TYPE_FACE
        meta.objClassId = 0
        
        obj = pyds.alloc_nvds_face_object()
        obj = DSFaceMessage.generate_face_meta(obj, image_link, ff_link, id, name, ts)
        meta.extMsg = obj
        meta.extMsgSize = sys.getsizeof(pyds.NvDsFaceObject)
        return meta

    def generate_face_meta(data, image_link, ff_link, id, name, ts):
        obj = pyds.NvDsFaceObject.cast(data)
        obj.age = 24
        obj.gender = str(id)
        obj.cap = str(ts)
        obj.hair = ""
        obj.glasses = ff_link 
        obj.facialhair = image_link
        obj.name = name
        obj.eyecolor = ""
        return obj
