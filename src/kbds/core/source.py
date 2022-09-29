"""
source description
:Author: InSun
:Version: v1.0-alpha
:Created: 2021-07-20
:Description:
"""
__all__ = ["Source", "Stream", "IPC"]


class Source(object):
    def __init__(self, id, **kwargs) -> None:
        """
        :param id: str, source id
        :param kwargs:
        """
        self.id = id

        # rt_ctx: dict, runtime context, is used to hold source runtime information
        self.rt_ctx = None

        name = kwargs.get('name')
        self.name = name if name and isinstance(name, str) else \
            str(id) + '-' + self.__class__.__name__

    def __str__(self) -> str:
        return "<%s %s %s> " % (self.__class__.__name__, self.id, self.name)


class Stream(Source):
    def __init__(self, id, uri, **kwargs) -> None:
        """
        :param id: str, source id
        :param uri: str, stream uri
        :param kwargs:
        """
        super().__init__(id, **kwargs)

        self.uri = uri
        self.regions = kwargs.get('regions')

    def set_index(self, index):
        self.idx = index

    def get_index(self):
        return self.idx


class IPC(Stream):
    def __init__(self, id, uri, ip, **kwargs) -> None:
        """
        :param id: str, source id
        :param uri: str, stream uri
        :param ip: str, camera ip address
        :param kwargs:
            username: str, camera username
            password: str, camera password
        """
        super().__init__(id, uri, **kwargs)
        
        self.ip = ip
        self.username = kwargs.get('username')
        self.password = kwargs.get('password')       
