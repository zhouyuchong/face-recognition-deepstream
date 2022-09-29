"""
source manager
:Author: InSun
:Version: v1.0-alpha
:Created: 2021-07-20
:Description:
"""
import functools
from threading import Lock

__all__ = ['SRCM']


def d_acquire_lock(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self._thread_safety:
            with self._lock:
                res = func(self, *args, **kwargs)
        else:
            res = func(self, *args, **kwargs)
        return res

    return wrapper


class SRCM:
    """
    source manager - simple version
    """
    _MIN_TIMEOUT = 1
    _MSG_ERR_SRC_ID_EXIST = "source id(%s) existed"
    _MSG_ERR_SRC_NOT_EXIST = "source id(%s) not exist"
    _MSG_ERR_SRC_NUM_EXCEED = "exceed max source num %d"
    _MSG_SUCCESS = "success"

    def __init__(self, thread_safety=False, max_src_num=float('inf')) -> None:
        """
        :param thread_safety: bool, whether srcm operation is thread safety
        :param max_src_count: int, max source num, default inf
        """
        self._max_src_num = max_src_num
        self._pool = {}
        self._thread_safety = thread_safety
        self._lock = Lock() if thread_safety else None
        self._available_index = [True] * max_src_num

    def exist(self, id):
        return id in self._pool

    @d_acquire_lock
    def add(self, src):
        """
        add source to srcm
        :param src: Source
        :return: (bool, string), result & message
        """
        if len(self._pool) >= self._max_src_num:
            return False, SRCM._MSG_ERR_SRC_NUM_EXCEED % self._max_src_num
        
        org_src = self._pool.setdefault(src.id, src)
        return (True, SRCM._MSG_SUCCESS) if org_src is src else \
            (False, SRCM._MSG_ERR_SRC_ID_EXIST % src.id)

    @d_acquire_lock
    def delete(self, id):
        """
        delete source from srcm
        :param id: str, source id
        :return: (bool, string, Source), result & message & source
        """
        pop_src = self._pool.pop(id, None)
        return (False, SRCM._MSG_ERR_SRC_NOT_EXIST % id, pop_src) \
            if pop_src is None else (True, SRCM._MSG_SUCCESS, pop_src)

    @d_acquire_lock
    def get(self, id: str = None):
        """
        get source list or
        get source with index
        :param id: source id
        :return: (bool, str, Source or List), result & message & source or list-of-source
        """
        if id:
            src = self._pool.get(id, None)
            return (True, SRCM._MSG_SUCCESS, src) if src else \
                (False, SRCM._MSG_ERR_SRC_NOT_EXIST % id, None)
        else:
            src_ls = []
            for _, v in self._pool.items():
                src_ls.append(v)

            return True, SRCM._MSG_SUCCESS, src_ls

    @d_acquire_lock
    def alloc_index(self, src):
        for i in range(self._max_src_num):
            if self._available_index[i] == True:
                self._available_index[i] = False
                src.set_index(i)
                print("set index of src {} to {}".format(src.id, i))
                return True
        print("fail to alloc index, maybe full.")
        return False

    @d_acquire_lock
    def clean_index(self, src):
        idx = src.get_index()
        self._available_index[idx] = True
        
    @d_acquire_lock
    def get_id_by_idx(self, index):
        for key in self._pool:
            if self._pool[key].idx == index:
                return key
        return None

    @property
    def total(self):
        return len(self._pool)


    