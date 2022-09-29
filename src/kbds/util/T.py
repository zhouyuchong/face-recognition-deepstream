"""
:Author: InSun
:Version: v1.0-alpha
:Created: 2021-07-20
:Description:
"""
import os

__all__ = ['sysp']


class SYSP:

    @property
    def pkg_path(self):
        import kbds
        return kbds.__path__

    @property
    def cfg_path(self):
        return self._second_path('configs')

    @property
    def lib_path(self):
        return self._second_path('libs')

    @property
    def model_path(self):
        return self._second_path('models')

    def _second_path(self, name):
        for path in self.pkg_path:
            cfg_path_s = os.path.join(path, name)
            if os.path.exists(cfg_path_s):
                return os.path.normpath(cfg_path_s)
        return ""       


sysp = SYSP()