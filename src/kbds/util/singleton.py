"""
:Author: InSun
:Version: v1.0-alpha
:Created: 2021-07-20
:Description:
"""
class Singleton:
    def __new__(cls, *args, **kwargs):        
        # TODO here has some bugs if use this with multiple inheritance
        # TODO Singleton maybe not thread safty
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls)
        return cls._instance