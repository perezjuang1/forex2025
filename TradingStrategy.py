from ConnectionFxcm import RobotConnection
from Price import RobotPrice
import datetime as dt
import time
import numpy as np


class Trading:
    def __init__(self, days, timeframe='m5', instrument="EUR/USD"):
        self.timeframe = timeframe
        self.instrument = instrument
        self.robotconnection = RobotConnection()
        self.connection = self.robotconnection.getConnection()
        self.corepy = self.robotconnection.getCorepy()
        self._robot_price = RobotPrice(days, self.instrument, self.timeframe)
        self.days = days

    def __del__(self):
        print('Object gets destroyed')
        self.connection.logout()

    def start_trade_monitor(self):    
        self.operation_detection(timeframe=self.timeframe)        
        while True:            
            currenttime = dt.datetime.now()  
            if self.timeframe == "m1" and currenttime.second == 0:
                self.operation_detection( timeframe=self.timeframe)    
                time.sleep(1)                    
            elif self.timeframe == "m5" and currenttime.second == 0 and currenttime.minute % 5 == 0:
                self.operation_detection( timeframe=self.timeframe)
                time.sleep(240)
            elif self.timeframe == "m15" and currenttime.second == 0 and currenttime.minute % 15 == 0:
                self.operationDetection( timeframe=self.timeframe)
                time.sleep(840)
            elif self.timeframe == "m30" and currenttime.second == 0 and currenttime.minute % 30 == 0:
                self.operationDetection( timeframe=self.timeframe)
                time.sleep(1740)
            elif self.timeframe == "H1" and currenttime.second == 0 and currenttime.minute == 0:
                self.operationDetection( timeframe=self.timeframe)
                time.sleep(3540)
            time.sleep(1)

    def operation_detection(self, timeframe): 
            df = self._robot_price.getPriceData(instrument=self.instrument, timeframe=timeframe, days=self.days, connection=self.connection)
            print("Price Data Received..." + str(timeframe) + " " + str(self.instrument) + " " + str(dt.datetime.now()))
            print("Price Data Length: " + str(len(df)) + " " + str(df))

# Example usage
if __name__ == "__main__":
    trading = Trading(days=10, timeframe='m5', instrument="EUR/USD")
    trading.start_trade_monitor()
