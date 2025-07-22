from ConnectionFxcm import RobotConnection
from Price import Price
import datetime as dt
import time
import numpy as np
import traceback
import threading
from ConfigurationOperation import ConfigurationOperation

class Trading:
    def __init__(self, days, timeframe=None, instrument="EUR/USD"):
        if timeframe is None:
            timeframe = ConfigurationOperation.timeframe
        self.timeframe = timeframe
        self.instrument = instrument
        self.robotconnection = RobotConnection()
        self.connection = self.robotconnection.getConnection()
        self.corepy = self.robotconnection.getCorepy()
        self._robot_price = Price(days, self.instrument, self.timeframe)
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
                self.operation_detection( timeframe=self.timeframe)
                time.sleep(840)
            elif self.timeframe == "m30" and currenttime.second == 0 and currenttime.minute % 30 == 0:
                self.operation_detection( timeframe=self.timeframe)
                time.sleep(1740)
            elif self.timeframe == "H1" and currenttime.second == 0 and currenttime.minute == 0:
                self.operation_detection( timeframe=self.timeframe)
                time.sleep(3540)
            time.sleep(1)

    def operation_detection(self, timeframe): 
        try:
            print(f"[LOG] Iniciando operación_detection - timeframe: {timeframe} - " + str(dt.datetime.now())  )
            df = self._robot_price.get_price_data(instrument=self.instrument, timeframe=timeframe, days=self.days, connection=self.connection)
        except Exception as e:
            print("Exception: " + str(e))
            print(traceback.format_exc())
            raise 

def run_trading_for_instrument(instrument):
    while True:
        trading = None
        try:
            time.sleep(5)
            trading = Trading(days=7, instrument=instrument)
            trading.start_trade_monitor()
        except Exception as e:
            print(f"Fatal error occurred for {instrument}. Restarting Trading session...")
            print(traceback.format_exc())
            del trading
            print("20 seconds to restart...")
            time.sleep(20)

if __name__ == "__main__":
    from ConfigurationOperation import ConfigurationOperation
    instruments = ConfigurationOperation.instruments
    threads = []
    for instrument in instruments:
        t = threading.Thread(target=run_trading_for_instrument, args=(instrument,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

