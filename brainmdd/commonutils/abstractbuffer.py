import threading
import time


class AbstractBuffer:
    def __init__(self, isFixedSize=False):
        self.__is_fixed_size = isFixedSize
        self.__isDataUpdated = False
    
    
    def isFixedSize(self):
        return self.__is_fixed_size


    def addData(self, sample):
        raise NotImplementedError


    def getData(self):
        raise NotImplementedError


    def isDataUpdated(self):
        return self.__isDataUpdated
    
    
    def setDataUpdatedFlag(self):
        self.__isDataUpdated = True
    
    
    def resetDataUpdatedFlag(self):
        self.__isDataUpdated = False