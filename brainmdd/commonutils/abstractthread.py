import threading

DEFAULT_FPS_MODE_FREQUENCY = 60     
DEFAULT_THREAD_PERIOD = 0.1         

class AbstractThread:

    def __init__(self):
        self.__threadStopEvent = threading.Event()
        self.__threadPeriod = DEFAULT_THREAD_PERIOD
        self.__threadHandler = threading.Thread(target=self.__run)


    def update(self): 
        raise NotImplementedError


    def __run(self):
        while not self.__threadStopEvent.is_set():
            self.update()
            self.__threadStopEvent.wait(self.__threadPeriod)


    def startThread(self):
        self.__threadHandler.start()


    def stopThread(self):
        self.__threadStopEvent.set()
        self.__threadHandler.join()


    def setThreadPeriod(self, period=0):
        if period > 0:
            self.__threadPeriod = period
            return True
        else:
            self.__threadPeriod = DEFAULT_THREAD_PERIOD
            return False
    
    
    def setThreadFrequency(self, frequency=0, fpsMode=False):
        if fpsMode:
            self.setThreadPeriod(1 / DEFAULT_FPS_MODE_FREQUENCY)
            return True
        if frequency > 0:
            self.setThreadPeriod(1/frequency)
            return True
        
        self.setThreadPeriod(DEFAULT_THREAD_PERIOD)
        return False


    def getThreadPeriod(self):
        return self.__threadPeriod
    
    
    def getThreadFrequency(self):
        return 1 / self.__threadPeriod