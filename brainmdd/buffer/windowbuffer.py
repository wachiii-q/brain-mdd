from brainmdd.commonutils.abstractbuffer import AbstractBuffer
import numpy as np


class WindowBuffer(AbstractBuffer):
    def __init__(self, numsOfChannel, numsOfSample):
        super().__init__(isFixedSize=True)
        self.__numsOfChannel = numsOfChannel            # debug purpose
        self.__numsOfSample = numsOfSample              # debug purpose
        self.__buffer = np.zeros((numsOfChannel, numsOfSample), dtype=np.float64)
        self.__isDataFormatUpdated = False


    def addData(self, sample):
        self.__buffer = np.roll(self.__buffer, -1, axis=1)
        self.__buffer[:,-1] = sample
        self.setDataUpdatedFlag()


    def addBatchData(self, sample):
        self.__buffer = np.array(sample)
        self.setDataUpdatedFlag()


    def getData(self):
        return self.__buffer


    def setDataFormat(self, dataFormat):
        self.__isDataFormatUpdated = True
        self.__dataFormat = dataFormat


    def getDataFormat(self, readonly=False):
        if not readonly:
            self.__isDataFormatUpdated = False
        return self.__dataFormat

    
    def isDataFormatUpdated(self):
        return self.__isDataFormatUpdated