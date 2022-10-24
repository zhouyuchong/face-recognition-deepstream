"""
deepstream pipeline abstract
:Author: InSun
:Version: v1.0-alpha
:Created: 2021-07-20
:Description:
"""
import abc
from ast import And
import functools
from telnetlib import PRAGMA_HEARTBEAT
from threading import RLock, Thread
from kafka import KafkaProducer
import json
import time
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

from kbds.util.singleton import Singleton
from kbds.util.file import *
from .srcm import SRCM
import kbds.util.constant as constant
from kbds.util.bus_kafka import init_bus_kafka

__all__ = ['DSPipeline']



Gst.init(None)

class DSPD:
    """
    deepstream pipeline decorator
    """
    lock = RLock()

    @staticmethod
    def d_acquire_lock(func):
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with DSPD.lock:
                res = func(*args, **kwargs)
            return res
        
        return wrapper


class DSPipeline(Singleton, abc.ABC):
    def __init__(self, conn_str="localhost;9090;deepstream", gpu_id=0, batch_size=32, batch_push_timeout=25000, 
            max_source_num=16) -> None:
        """
        deepstream pipeline
        :param gpu_id: int, stream-mux gup_id
        :param batch_size: int, stream-mux batch_size
        :param batch_push_timeout: int, stream-mux batch_push_timeout
        :param max_source_num: int, pipeline supported max source num
        """
        super().__init__()
        self.srcm = SRCM(max_src_num=max_source_num)
        self.is_first_src = True

        self.gpu_id = gpu_id
        self.max_source_num = max_source_num
        self.batch_size = max_source_num if batch_size < max_source_num else batch_size
        self.batch_push_timeout = batch_push_timeout

        # pipeline element
        self.pipeline = None
        self.streammux = None
        self.msgconv = None
        self.msgbroker = None
        self.nvanalytics = None
        self.loop = GLib.MainLoop()
        self.thread = None
        self.producer = init_bus_kafka(config=conn_str)

        # system control
        self._init = False

    @abc.abstractmethod
    def build_pipeline_abs(self, pipeline, streammux, msgconv, msgbroker):
        """
        build pipeline abstract method
        build deepstream pipeline here
        element streammux has already add into pipeline

        :param pipeline: Gst.Pipeline, stream pipeline
        :param streammux: Gst.Element, nvstreammux
        :return: (bool, message), result & message
        """
        ...

    def update_src_abs(self, src, data):
        """
        if ur pipeline support update source config, re-implement this function, default not

        :raise NotImplementedError
        :param src:
        :param data:
        :return:
        """
        # TODO log here
        print("source %s(id) update infrmation:\n" % src.id, data)
        raise NotImplementedError("current app does not support update source information")

    def bus_call_abs(self, bus, message):
        t = message.type
        # TODO log here
        if t == Gst.MessageType.EOS:
            print("end of stream\n")
            self.loop.quit() 
        elif t==Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            print("Warning: %s: %s\n" % (err, debug))
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            struct = str(message.get_structure())
            search_index = re.search("source-bin-[0-9]{0,1}/", struct)
            search_index = search_index[0].split('-')[2][:-1]
            err_id = self.srcm.get_id_by_idx(int(search_index))
            try:
                self.del_src(id=err_id)
                print("id: {},  error: {}, delete.".format(err_id, err))
                time_local = time.localtime(int(time.time()))
                dt = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
                msg = {"time": dt, "id": str(err_id), "message":str(err)}
                msg = json.dumps(msg).encode('utf-8')
                
                if str(err).startswith("gst-resource-error-quark"):
                    # resource errors from 1 to 16
                    if str(err).endswith("(9)"):
                        # handle_read_error()
                        print(err)
                        # print("Error: %s: %s\n" % (err, debug))
                    else:
                        self.producer.send('error', msg)
            except Exception:
                print("id: {} encounters error: {}, delete.".format(err_id, err))
        elif t == Gst.MessageType.ELEMENT:
            struct = message.get_structure()
            #Check for stream-eos message
            if struct is not None and struct.has_name("stream-eos"):
                parsed, stream_id = struct.get_uint("stream-id")
                if parsed:
                    #Set eos status of stream to True, to be deleted in delete-sources
                    print("Got EOS from stream %d" % stream_id)
        return True 

    @DSPD.d_acquire_lock
    def init(self):
        """
        init pipeline
        :return: (bool, str), result & message
        """
        if self._init:
            return True, "system already init"

        self.pipeline = Gst.Pipeline()
        if not self.pipeline:
            return False, "unable to create pipeline"

        self.streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
        if not self.streammux:
            return False, "unable to create nvstreammux"

        self.msgconv = Gst.ElementFactory.make("nvmsgconv", "nvmsg-converter")
        if not self.msgconv:
            return False, "Unable to create msgconv \n"

        self.msgbroker = Gst.ElementFactory.make("nvmsgbroker", "nvmsg-broker")
        if not self.msgbroker:
            return False, " Unable to create msgbroker \n"

        self.nvanalytics = Gst.ElementFactory.make("nvdsanalytics", "analytics")
        if not self.nvanalytics:
            return False, " Unable to create nvanalytics \n"

        self.streammux.set_property("batched_push_timeout", self.batch_push_timeout)
        self.streammux.set_property("batch_size", self.batch_size)
        self.streammux.set_property("gpu_id", self.gpu_id)
        self.streammux.set_property("live-source", 1)

        self.pipeline.add(self.streammux)

        # build different pipeline here
        ret, msg = self.build_pipeline_abs(self.pipeline, self.streammux, self.msgconv, self.msgbroker, self.nvanalytics)
        if not ret:
            return ret, msg

        # connect msg bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call_abs, )

        self._init = True
        return ret, "success"


    @DSPD.d_acquire_lock
    def start(self):
        """
        start the pipeline
        :return: (bool, str), result & message
        """
        # TODO log here
        ret, msg = self.init()
        if not ret:
            return ret, "pipeline init failed, %s" % msg

        _, s, ps = self.pipeline.get_state(0.0)
        if s == Gst.State.PLAYING or ps == Gst.State.PLAYING:
            return True, "pipeline is running, state %s, pending state %s" % (s, ps)

        # run message loop
        self.thread = Thread(target=self._msg_thread_func)
        self.thread.start()
        
        ret = self.pipeline.set_state(Gst.State.NULL)
        if ret == Gst.StateChangeReturn.SUCCESS:
            return True, "success, state %s" % ret
        elif ret == Gst.StateChangeReturn.FAILURE:
            # TODO log here
            return False, "start pipeline failed"
        elif ret == Gst.StateChangeReturn.ASYNC:
            return True, "success, state %s" % ret

    @DSPD.d_acquire_lock
    def stop(self):
        """
        stop the pipeline
        :return: (bool, str), result & message
        """
        # ret, _, src_ls = self.srcm.get()
        # if not ret:
        #     return False, "get src info failed"
        # for src in src_ls:
        #     src.rt_ctx['bin'].set_state(Gst.State.NULL)
        #     src.rt_ctx['enable'] = False

        self.pipeline.set_state(Gst.State.NULL)
        self.producer.close()
        if self.thread is not None and self.thread.is_alive():
            self.loop.quit()
            self.thread = None

        return True, "success"

    @DSPD.d_acquire_lock
    def get_src(self, id=None):
        """
        get source list or
        get source with index
        :param id: str, source id
        :return: (bool, str, Source or List), result & message & source or list-of-source
        """
        return self.srcm.get(id=id)

    @DSPD.d_acquire_lock
    def add_src(self, src):
        """
        :param src: Source
        :return: (bool, str)
        """
        # make not exceed max source num
        if self.srcm.total >= self.max_source_num:
            return False, "exceed max source num %d" % self.max_source_num

        if self.srcm.exist(src.id):
            return False, "source id %s exist" % src.id
        
        # make sure the pipeline is start
        _, s, _ = self.pipeline.get_state(0.0)
        if s == Gst.State.NULL and self.is_first_src is not True:
            return False, "pipeline is stoped"
        # print("pipeline status:", s)

        self.srcm.alloc_index(src)
        # create source bin
        source_bin, msg = self._create_uridecode_bin(src.idx, src.uri)
        if not source_bin:
            return False, msg
        self.pipeline.add(source_bin)

        # add runtime infotmation
        src.rt_ctx = {'enable': True,
            'bin': source_bin,
            'bin_name': self._uridecode_bin_name(src.idx)}

        # playing source
        state_return = source_bin.set_state(Gst.State.PLAYING)
        print("add set return :", state_return)
        if state_return == Gst.StateChangeReturn.SUCCESS:
            print("source state change  success")
        elif state_return == Gst.StateChangeReturn.FAILURE:
            print("error, source state change failure")

        # add src to source manager
        ret, msg = self.srcm.add(src)
        if not ret:
            raise Exception("add source to srm failed")

        if self.is_first_src:
            self.is_first_src = False
            print("Starting pipeline \n")
            state_return  = self.pipeline.set_state(Gst.State.PLAYING)

        return True, "success"

    @DSPD.d_acquire_lock
    def del_src(self, id):
        """
        :param id: str, source id
        :return: (bool, str, Source), result & message & deleted source
        """
        # print("total:", self.srcm.total)
        

            # print(state_return)

        # get src
        ret, msg, src = self.srcm.get(id)
        if not ret:
            return ret, msg, None
        
        if self.srcm.total == 1:
            self.streammux.set_state(Gst.State.NULL)
            state_return = self.pipeline.set_state(Gst.State.NULL)
            self.is_first_src = True

        source_bin = src.rt_ctx['bin']
        state_return = source_bin.set_state(Gst.State.NULL)

        if state_return == Gst.StateChangeReturn.FAILURE:
            return False, "source bin stop failed", src
        elif state_return == Gst.StateChangeReturn.ASYNC:
            state_return = source_bin.get_state(Gst.CLOCK_TIME_NONE)

        pad_name = "sink_%s" % src.idx
        sinkpad = self.streammux.get_static_pad(pad_name)
        if sinkpad is not None:
            sinkpad.send_event(Gst.Event.new_flush_stop(False))
            self.streammux.release_request_pad(sinkpad)
        print("delete finished")

        self.pipeline.remove(source_bin)

        src.rt_ctx = None
        self.srcm.clean_index(src)
        ret, msg, src = self.srcm.delete(id)    
        if not ret:
            raise Exception("srcm delete source failed")

        return ret, msg, src        

    @DSPD.d_acquire_lock
    def pause_src(self, id):
        """
        :param id: str, source id
        :return: (bool, str), result & message
        """ 
        raise NotImplementedError

        ret, msg, src = self.srcm.get(id)
        if not ret:
            return ret, msg

        source_bin = src.rt_ctx["bin"]
        state_return = source_bin.set_state(Gst.State.PAUSED)
        if state_return == Gst.StateChangeReturn.FAILURE:
            return False, "source %s change state failure %s" % (src.id, state_return)
        
        src.rt_ctx['enable'] = False
        return True, "success"

    @DSPD.d_acquire_lock
    def play_src(self, id):
        """

        :param id: str, source id
        :return: (bool, str), result & message
        """
        
        ret, msg, src = self.srcm.get(id=id)
        print("src:", src)
        if not ret or isinstance(src, list):
            return ret, msg
        source_bin = src.rt_ctx["bin"]
        print("source_bin:",source_bin)
        state_return = source_bin.set_state(Gst.State.PLAYING)
        print(state_return)
        if state_return == Gst.StateChangeReturn.FAILURE:
            return False, "source %s change state failure %s" % (src.id, state_return)

        src.rt_ctx['enable'] = True
        return True, "success"

    @DSPD.d_acquire_lock
    def update_src(self, id, data):
        """

        :param id: str, source id
        :param data: any, source update information
        :return: (bool, str), result & message
        """
        ret, msg, src = self.srcm.get(id)
        if not ret:
            return ret, msg

        return self.update_src_abs(src, data)

    @DSPD.d_acquire_lock
    def set_analytics(self, id, type, data):
        """
        :param id:str, source id
        :param type: enum, analytics type
        :param data: dict, line crossing data
        :return: (bool, str), result & path of analytics
        """
        ret, msg, src = self.srcm.get(id)
        if not ret:
            return ret, msg
        for key, value in data.items():
            coors = value.split(';')            
            for i in range(len(coors)):
                if i == 0:
                    coors[i] = int(float(coors[i]) * constant.MUXER_OUTPUT_WIDTH) 
                    continue
                elif i == 1:
                    coors[i] = int(float(coors[i]) * constant.MUXER_OUTPUT_HEIGHT) 
                    continue
                elif i%2 == 0:
                    coors[i] = int(float(coors[i]) * constant.MUXER_OUTPUT_WIDTH) 
                else:
                    coors[i] = int(float(coors[i]) * constant.MUXER_OUTPUT_HEIGHT) 
            coors = [str(coor) for coor in coors]
            semicolon = ';'
            coors = semicolon.join(coors)
            data[key] = coors
        print(data)
        idx = src.get_index()
        if type == 1:
            if modify_analytics_crossingline(constant.ANALYTICS_CONFIG_FILE, max_source_number=self.max_source_num, index=idx, enable=1, extended=0, mode='balanced', class_id=2, **data):
                self.nvanalytics.set_property("config-file", constant.ANALYTICS_CONFIG_FILE)      
            return True, self.nvanalytics.get_property("config-file")
        elif type == 2:
            if modify_analytics_ROI(constant.ANALYTICS_CONFIG_FILE, max_source_number=self.max_source_num, index=idx, enable=1, inverse_roi=0, class_id=-1, **data):
                self.nvanalytics.set_property("config-file", constant.ANALYTICS_CONFIG_FILE)      
            return True, self.nvanalytics.get_property("config-file")
        else:
            return False, "None"
        


    def cb_decodebin_newpad(self, bin, pad, data) -> None:
        """
        callback function
        decodebin pad_added signal
        :param bin:
        :param pad:
        :param data: str, source id
        :return:
        """
        # TODO log here
        caps = pad.get_current_caps()
        gststruct = caps.get_structure(0)
        gstname = gststruct.get_name()
        print("gstname=",gstname)

        if gstname.find("video") != -1:
            pad_name = "sink_%s" % data
            # print(pad_name)
            # get a sink pad from the streammux, link to decodebin
            print("pad name: ", pad_name)
            sinkpad = self.streammux.get_request_pad(pad_name)
            # TODO log here
            if pad.link(sinkpad) == Gst.PadLinkReturn.OK:
                print("Decodebin linked to pipeline")
            else:
                print("Failed to link decodebin to pipeline\n")

    def cb_decodebin_child_added(self, child_proxy, obj, name, data) -> None:
        """

        :param child_proxy:
        :param obj:
        :param name:
        :param data:
        :return:
        """
        # TODO log here
        if name.find("decodebin") != -1:
            obj.connect("child-added", self.cb_decodebin_child_added, data)

        if name.find("nvv4l2decoder") != -1:
            obj.set_property("gpu_id", self.gpu_id)

    def _uridecode_bin_name(self, bin_id):
        return "source-bin-%s" % bin_id

    def _create_uridecode_bin(self, bin_id, uri):
        bin_name = self._uridecode_bin_name(bin_id)
        bin = Gst.ElementFactory.make("uridecodebin", bin_name)
        if not bin:
            return None, "failed to create uridecodebin"

        bin.set_property("uri", uri)
        bin.connect("pad_added", self.cb_decodebin_newpad, bin_id)
        bin.connect("child-added", self.cb_decodebin_child_added, bin_id)

        return bin, "success"

    def _msg_thread_func(self):
        try:
            self.loop.run()
        except Exception as e:
            print(e)

