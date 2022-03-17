import sys
sys.path.append('../')
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
from gi.repository import GLib
import pyds

import utils.functions
import numpy as np
from common.utils import long_to_uint64
import json
import base64

import Gstpipeline.message

PRIMARY_GIE = 1
SECONDARY_GIE = 2
TERTIARY = 3

RFACE_THRESHOLD = 0.97

YOLO_PERSON_TRACK_ID = []
RFACE_TRACK_ID = []
AFACE_CAL_ID = []
CROPED_FACE = []
face_count = 0

fps_streams={}


def pic_crop_probe(pad,info,u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ON SGIE ")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    global YOLO_PERSON_TRACK_ID
    global RFACE_TRACK_ID
    global CROPED_FACE
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

        frame_number=frame_meta.frame_num
        l_obj=frame_meta.obj_meta_list
        Rface2Aface_pass_signal = False
        while l_obj is not None:
            try: 
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            # obj_counter[obj_meta.class_id] += 1
            if obj_meta.unique_component_id == SECONDARY_GIE and obj_meta.class_id == 0 and obj_meta.object_id not in CROPED_FACE and obj_meta.object_id in RFACE_TRACK_ID:
                CROPED_FACE.append(obj_meta.object_id)
                print(CROPED_FACE)
                # print(CROPED_FACE)
                n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                n_frame = utils.functions.crop_object(n_frame, obj_meta)
                # convert python array into numpy array format in the copy mode.
                frame_copy = np.array(n_frame, copy=True, order='C')
                # print(frame_copy.shape)
                shape_0 = frame_copy.shape[0]
                shape_1 = frame_copy.shape[1]
                # convert the array into cv2 default color format
                # frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGRA)
                # print(frame_copy)
                # img_path = "{}/stream_{}/tracked_face_{}.jpg".format(folder_name, frame_meta.pad_index, obj_meta.object_id)
             
                # print(obj_meta.object_id, obj_meta.confidence, "------------->save")
                # cv2.imwrite(img_path, frame_copy)
                msg_meta = pyds.alloc_nvds_event_msg_meta()
                msg_meta.bbox.top = obj_meta.rect_params.top
                msg_meta.bbox.left = obj_meta.rect_params.left
                msg_meta.bbox.width = obj_meta.rect_params.width
                msg_meta.bbox.height = obj_meta.rect_params.height
                msg_meta.frameId = frame_number
                msg_meta.trackingId = long_to_uint64(obj_meta.object_id)
                msg_meta.confidence = obj_meta.confidence
                lists = frame_copy.tolist()
                json_str = json.dumps(lists)
                base64array = str(base64.b64encode(json_str.encode('utf-8')),"utf-8")
                # pickle_str = pickle.dumps(frame_copy)
                msg_meta = Gstpipeline.message.generate_event_msg_meta(msg_meta, base64array)
                user_event_meta = pyds.nvds_acquire_user_meta_from_pool(
                    batch_meta)
                if user_event_meta:
                    user_event_meta.user_meta_data = msg_meta
                    user_event_meta.base_meta.meta_type = pyds.NvDsMetaType.NVDS_EVENT_MSG_META
                    # Setting callbacks in the event msg meta. The bindings
                    # layer will wrap these callables in C functions.
                    # Currently only one set of callbacks is supported.
                    pyds.user_copyfunc(user_event_meta, Gstpipeline.message.meta_copy_func)
                    pyds.user_releasefunc(user_event_meta, Gstpipeline.message.meta_free_func)
                    pyds.nvds_add_user_meta_to_frame(frame_meta,
                                                     user_event_meta)
                else:
                    print("Error in attaching event meta to buffer\n")
                                                                
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break

        try:
            l_frame=l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

def sgie_src_pad_buffer_probe(pad,info,u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ON SGIE ")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    global YOLO_PERSON_TRACK_ID
    global RFACE_TRACK_ID
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

        frame_number=frame_meta.frame_num
        l_obj=frame_meta.obj_meta_list
        Rface2Aface_pass_signal = False
        while l_obj is not None:
            try: 
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            # obj_counter[obj_meta.class_id] += 1
            if obj_meta.unique_component_id == SECONDARY_GIE and obj_meta.confidence > RFACE_THRESHOLD:
                parent = obj_meta.parent
                # print(parent.unique_component_id, parent.class_id)
                if parent is None:
                    print("THIS AN ANONYMOUS FACE, DISCARD")
                    return Gst.PadProbeReturn.DROP
                if parent.object_id in YOLO_PERSON_TRACK_ID:
                    return Gst.PadProbeReturn.DROP
                if parent.object_id not in YOLO_PERSON_TRACK_ID:
                    YOLO_PERSON_TRACK_ID.append(parent.object_id)
                    global face_count
                    obj_meta.object_id = face_count
                    RFACE_TRACK_ID.append(face_count)
                    face_count += 1
                                                                
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break

        try:
            l_frame=l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

def tgie_src_pad_buffer_probe(pad,info,u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ON tgie ")
        return

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

        frame_number=frame_meta.frame_num
        l_obj=frame_meta.obj_meta_list
        global FACE_ALL
        
        while l_obj is not None:
            try: 
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            if obj_meta.object_id in RFACE_TRACK_ID and obj_meta.object_id not in AFACE_CAL_ID:
                AFACE_CAL_ID.append(obj_meta.object_id)

                l_user = obj_meta.obj_user_meta_list
                while l_user is not None:
                    try:
                        # Note that l_user.data needs a cast to pyds.NvDsUserMeta
                        # The casting is done by pyds.NvDsUserMeta.cast()
                        # The casting also keeps ownership of the underlying memory
                        # in the C code, so the Python garbage collector will leave
                        # it alone
                        user_meta=pyds.NvDsUserMeta.cast(l_user.data) 
                    except StopIteration:
                        break
                    
                    # Check data type of user_meta 
                    if(user_meta and user_meta.base_meta.meta_type==pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META): 
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
                        '''print(obj_meta.object_id, obj_meta.confidence)

                        msg_meta = pyds.alloc_nvds_event_msg_meta()
                        msg_meta.bbox.top = obj_meta.rect_params.top
                        msg_meta.bbox.left = obj_meta.rect_params.left
                        msg_meta.bbox.width = obj_meta.rect_params.width
                        msg_meta.bbox.height = obj_meta.rect_params.height
                        msg_meta.frameId = frame_number
                        msg_meta.trackingId = long_to_uint64(obj_meta.object_id)
                        msg_meta.confidence = obj_meta.confidence
                        lists = normal_array.tolist()
                        json_str = json.dumps(lists)
                        # pickle_str = pickle.dumps(frame_copy)
                        msg_meta = Gstpipeline.message.generate_event_msg_meta(msg_meta, json_str)
                        user_event_meta = pyds.nvds_acquire_user_meta_from_pool(
                            batch_meta)
                        if user_event_meta:
                            user_event_meta.user_meta_data = msg_meta
                            user_event_meta.base_meta.meta_type = pyds.NvDsMetaType.NVDS_EVENT_MSG_META
                            # Setting callbacks in the event msg meta. The bindings
                            # layer will wrap these callables in C functions.
                            # Currently only one set of callbacks is supported.
                            pyds.user_copyfunc(user_event_meta, Gstpipeline.message.meta_copy_func)
                            pyds.user_releasefunc(user_event_meta, Gstpipeline.message.meta_free_func)
                            pyds.nvds_add_user_meta_to_frame(frame_meta,
                                                            user_event_meta)
                        else:
                            print("Error in attaching event meta to buffer\n")'''
                    
                        try:
                            l_user=l_user.next
                        except StopIteration:
                            break            
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break
        
        try:
            l_frame=l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

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
        global fps_streams
        fps_streams["stream{}".format(frame_meta.pad_index)].get_fps()
        try:
            l_frame=l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK
